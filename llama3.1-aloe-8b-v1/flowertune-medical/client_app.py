"""mmfl: A Flower / FlowerTune app.

클라이언트 측 학습 루프(ClientApp)를 정의합니다.
 - 각 클라이언트 파티션 데이터 로딩 및 전처리
 - 서버에서 전달받은 전역 가중치로 로컬 모델 초기화
 - 라운드별 코사인 스케줄에 따른 학습률 적용 후 SFTTrainer 학습
 - 업데이트된 PEFT 가중치와 훈련 메트릭을 서버로 전송
"""

import logging
import os
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from mmfl.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
)
from mmfl.models import cosine_annealing, get_model
from mmfl.config_utils import override_config_with_env_vars, replace_keys

logging.basicConfig(level=logging.INFO)

# 경고 메시지 억제(토크나이저 병렬 경고 및 Ray CPU 경고)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# 위와 동일(일부 환경에서 중복 정의가 안전함)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp 인스턴스 생성
app = ClientApp()


def _resolve_task_name(cfg: DictConfig) -> str:
    task_cfg = getattr(cfg, "task", None)
    name = getattr(task_cfg, "name", None) if task_cfg is not None else None
    return name or "general-nlp"


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data.

    서버에서 전달된 설정과 가중치를 바탕으로 로컬 데이터에 대해 한 라운드 학습을 수행합니다.
    """
    # 설정 읽기 및 환경변수로 덮어쓰기
    run_config = dict(context.run_config)
    run_config = override_config_with_env_vars(run_config, "MMFL_APP_CONFIG_")

    # 실행/노드 설정 파싱
    # 1) 기본: Server/SuperNode가 제공한 node_config 사용
    # 2) 폴백: 환경변수로 직접 지정 가능(MMFL_PARTITION_ID, MMFL_NUM_PARTITIONS)
    #    - K8s 파드 배포 등 네트워크 모드에서 간편 제어 용도
    partition_id = int(
        os.environ.get(
            "MMFL_PARTITION_ID",
            context.node_config.get("partition-id", 0),
        )
    )
    num_partitions = int(
        os.environ.get(
            "MMFL_NUM_PARTITIONS",
            context.node_config.get("num-partitions", 1),
        )
    )
    num_rounds = run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)
    trust_remote_code = bool(getattr(cfg.model, "trust_remote_code", False))
    task_name = _resolve_task_name(cfg)

    cuda_available = torch.cuda.is_available()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    logging.info(
        "ClientApp partition=%s cuda_available=%s device_count=%s CUDA_VISIBLE_DEVICES=%s",
        partition_id,
        cuda_available,
        torch.cuda.device_count(),
        visible_devices,
    )

    # 현재 클라이언트에 할당된 데이터 파티션 로딩
    validation_ratio = getattr(cfg.train, "validation_split_ratio", None)
    validation_seed = getattr(cfg.train, "validation_split_seed", None)

    trainset = load_data(
        partition_id,
        num_partitions,
        cfg.static.dataset.name,
        split="train",
        eval_split_ratio=validation_ratio,
        eval_split_seed=validation_seed,
        llm_task=task_name,
    )
    if trainset is None:
        raise RuntimeError(
            f"No training data available for partition {partition_id}"
        )
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(
        cfg.model.name, trust_remote_code=trust_remote_code, llm_task=task_name
    )

    # 모델을 로드하고 서버에서 받은 전역 가중치로 초기화
    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    # 현재 라운드에 사용할 학습률을 코사인 스케줄로 계산
    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]  # 체크포인트 저장 경로

    # TRL SFTTrainer 구성(포맷팅 함수/콜레이터 포함)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    # 로컬 학습 진행(한 라운드)
    results = trainer.train()

    # 학습 결과를 서버로 반환할 메시지 구성(업데이트된 가중치/메트릭)
    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on the local validation split."""
    # 설정 읽기 및 환경변수로 덮어쓰기
    run_config = dict(context.run_config)
    run_config = override_config_with_env_vars(run_config, "MMFL_APP_CONFIG_")

    partition_id = int(
        os.environ.get(
            "MMFL_PARTITION_ID",
            context.node_config.get("partition-id", 0),
        )
    )
    num_partitions = int(
        os.environ.get(
            "MMFL_NUM_PARTITIONS",
            context.node_config.get("num-partitions", 1),
        )
    )
    cfg = DictConfig(replace_keys(unflatten_dict(run_config)))
    trust_remote_code = bool(getattr(cfg.model, "trust_remote_code", False))
    task_name = _resolve_task_name(cfg)

    validation_ratio = getattr(cfg.train, "validation_split_ratio", None)
    validation_seed = getattr(cfg.train, "validation_split_seed", None)

    evalset = load_data(
        partition_id,
        num_partitions,
        cfg.static.dataset.name,
        split="eval",
        eval_split_ratio=validation_ratio,
        eval_split_seed=validation_seed,
        llm_task=task_name,
    )
    if evalset is None or len(evalset) == 0:
        logging.info(
            "Client-side evaluation skipped (partition=%s has no validation data).",
            partition_id,
        )
        metrics = {"num-examples": 1, "skipped": 1}
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)

    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(
        cfg.model.name, trust_remote_code=trust_remote_code, llm_task=task_name
    )

    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    training_arguments = TrainingArguments(**cfg.train.training_arguments)
    training_arguments.output_dir = msg.content["config"]["save_path"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        eval_dataset=evalset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    metrics_dict = trainer.evaluate()
    eval_loss = float(metrics_dict.get("eval_loss", float("nan")))
    metrics = {
        "eval_loss": eval_loss,
        "loss": eval_loss,
        "num-examples": len(evalset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
