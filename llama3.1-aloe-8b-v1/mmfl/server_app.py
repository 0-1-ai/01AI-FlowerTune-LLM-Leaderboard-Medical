"""mmfl: A Flower / FlowerTune app.

서버 측 엔트리 포인트(ServerApp)를 정의합니다.
 - 초기 전역(글로벌) 모델 가중치를 준비
 - 커스텀 FedAvg 전략 설정 및 실행
 - 학습 라운드마다(설정된 주기) 전역 모델 체크포인트 저장
"""

import logging
import os
from datetime import datetime

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import gc
import torch

from mmfl.models import get_model
from mmfl.strategy import FlowerTuneLlm, FlexLoraStrategy
from mmfl.config_utils import override_config_with_env_vars, replace_keys


LOGGER = logging.getLogger(__name__)

def _use_cpu_for_server_save() -> bool:
    """Return True if server checkpoint saving should happen on CPU."""
    return os.getenv("MMFL_SERVER_SAVE_ON_CPU", "0").lower() in {"1", "true", "yes"}


# Create ServerApp
app = ServerApp()  # Flower 서버 애플리케이션 인스턴스 생성


def _server_eval_disabled() -> bool:
    """Return True if server-side evaluation/saving should be skipped entirely."""
    return os.getenv("MMFL_SERVER_DISABLE_EVAL", "0").lower() in {"1", "true", "yes"}


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp.

    서버 실행 시 최초 한 번 호출되어 전체 학습을 오케스트레이션합니다.
    """
    # 현재 시각 기준으로 결과(체크포인트)를 저장할 디렉터리 생성
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # pyproject.toml 설정 읽기 및 환경변수로 덮어쓰기
    run_config = dict(context.run_config)  # 수정 가능한 복사본 생성
    run_config = override_config_with_env_vars(run_config, "MMFL_APP_CONFIG_")

    num_rounds = run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(run_config)))

    # 초기 전역(글로벌) 모델 가중치 준비
    init_model = get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))
    # 초기 모델의 GPU 메모리를 해제해 이후 평가 단계에서 중복 로딩으로 인한 OOM 방지
    del init_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 커스텀 FedAvg/FlexLoRA 전략 정의
    use_flexlora = bool(getattr(cfg.strategy, "use_flexlora", False))
    strategy_cls = FlexLoraStrategy if use_flexlora else FlowerTuneLlm
    strategy = strategy_cls(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
    )

    # 전략을 시작하고 FedAvg를 num_rounds 동안 실행
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        # 클라이언트에 전달할 훈련 관련 설정(여기선 저장 경로만 사용)
        train_config=ConfigRecord({"save_path": save_path}),
        evaluate_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )


# 전략 내부에서 호출할 평가 함수를 생성합니다.
# 여기서는 평가 대신, 주기적으로 전역 PEFT 모델 체크포인트를 저장합니다.
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model.

    실제 메트릭 산출 대신, 서버 라운드 번호에 따라 전역 모델을 저장합니다.
    저장 주기와 최종 라운드는 실행 설정으로부터 주어집니다.
    """

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # 환경 변수로 서버 평가/저장을 완전히 비활성화
        if _server_eval_disabled():
            return MetricRecord()
        # 서버 라운드가 0이 아니고, 저장 주기에 해당하면 모델 저장
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            quant_setting = getattr(model_cfg, "quantization", 4)
            quant_requires_cuda = quant_setting in (4, 8)
            force_cpu_save = _use_cpu_for_server_save()
            load_on_gpu = quant_requires_cuda and not force_cpu_save

            if load_on_gpu and not torch.cuda.is_available():
                LOGGER.warning(
                    "Skipping checkpoint save on round %s because model '%s' "
                    "requires CUDA for %s-bit quantization but no GPU is available.",
                    server_round,
                    getattr(model_cfg, "name", "unknown"),
                    quant_setting,
                )
                return MetricRecord()

            # 동일한 구성으로 모델을 초기화하고, 집계된 가중치를 주입
            model = get_model(model_cfg, force_cpu=not load_on_gpu)
            set_peft_model_state_dict(model, arrays.to_torch_state_dict())

            # 저장 경로: results/<timestamp>/peft_<round>
            model.save_pretrained(f"{save_path}/peft_{server_round}")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return MetricRecord()

    return evaluate
