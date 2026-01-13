"""mmfl: A Flower / FlowerTune app.

Flower Datasets를 이용해 데이터셋을 로딩/파티셔닝하고, 프롬프트 포맷팅과
토크나이저/데이터 콜레이터를 제공합니다.
"""

from typing import Dict, Optional, Tuple

from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

FDS = None  # FederatedDataset 캐시(다시 생성 방지)
# partition_id, dataset_name, num_partitions, eval_ratio, seed, task, split -> Dataset
PARTITION_CACHE: Dict[Tuple[int, str, int, float, int, str, str], Optional[Dataset]] = {}
DEFAULT_TASK_NAME = "general-nlp"


def _normalize_task_name(task_name: Optional[str]) -> str:
    """Normalize task identifier to a canonical string."""
    if not task_name:
        return "generalnlp"
    return task_name.replace("-", "").lower()


def formatting_prompts_func(example):
    """Construct prompts.

    Alpaca 스타일 프롬프트를 구성합니다. instruction/response 쌍을 하나의 텍스트로 병합합니다.
    """
    output_texts = []
    # Constructing a standard Alpaca
    # (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    for i in range(len(example["instruction"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n"
            f"### Response: {example['response'][i]}"
        )
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_qwen(example):
    """Construct prompts for Qwen (ChatML format)."""
    output_texts = []
    for i in range(len(example["instruction"])):
        # ChatML format without strong code system prompt
        text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['response'][i]}<|im_end|>"
        )
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_code_qwen(example):
    """Construct prompts for Qwen (ChatML format) with strict code-only constraint."""
    output_texts = []
    # System prompt to enforce code-only output, aligning with HumanEval style
    system_content = (
        "You are a coding expert. Write a response that appropriately completes the request. "
        "Provide ONLY the executable code. Do not include explanations, markdown formatting, or comments unless part of the code."
    )
    
    for i in range(len(example["instruction"])):
        instruction = example["instruction"][i]
        input_val = example.get("input", [""] * len(example["instruction"]))[i] # Handle input if exists
        
        user_content = f"{instruction}\n{input_val}".strip()
        
        # ChatML format
        text = (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['response'][i]}<|im_end|>"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(
    model_name: str, trust_remote_code: bool = False, llm_task: Optional[str] = None
):
    """Get tokenizer, data_collator and prompt formatting.

    - 토크나이저: EOS를 패딩 토큰으로 사용하고 오른쪽 패딩
    - 콜레이터: 응답 부분만 로스에 기여하도록 마스킹(DataCollatorForCompletionOnlyLM)
    - 프롬프트 포맷팅 함수 반환
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
        trust_remote_code=trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Normalize task name for comparison
    normalized_task = _normalize_task_name(llm_task) if llm_task else DEFAULT_TASK_NAME

    # Check if model is Qwen
    if "qwen" in model_name.lower():
        response_template_with_context = "<|im_start|>assistant\n"
        response_template_ids = tokenizer.encode(
            response_template_with_context, add_special_tokens=False
        )
        if normalized_task == "code":
            formatting_func = formatting_prompts_func_code_qwen
        else:
            formatting_func = formatting_prompts_func_qwen
    else:
        response_template_with_context = "\n### Response:"  # alpaca response tag
        formatting_func = formatting_prompts_func
        # For Alpaca, slice the template IDs (to remove leading tokens like BOS)
        response_template_ids = tokenizer.encode(
            response_template_with_context, add_special_tokens=False
        )[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_func


def formatting(dataset):
    """Format dataset.

    일부 태스크에서 input을 instruction 뒤에 공백으로 이어 붙여 단일 instruction으로 사용합니다.
    """
    dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
    return dataset


def reformat(dataset, llm_task):
    """Reformat datasets.

    원본 컬럼을 모델 학습에 맞게 재구성합니다.
    태스크 유형에 따라 컬럼 추가/제거가 다릅니다.
    """
    task_key = _normalize_task_name(llm_task)
    dataset = dataset.rename_column("output", "response")
    if task_key in ["finance", "code"]:
        dataset = dataset.map(formatting, remove_columns=["input"])
    if task_key == "medical":
        dataset = dataset.remove_columns(["instruction"])
        dataset = dataset.rename_column("input", "instruction")
    return dataset


def _populate_partition_cache(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    eval_split_ratio: float,
    eval_split_seed: int,
    llm_task: Optional[str],
) -> None:
    """Load dataset partition and populate train/eval cache entries."""
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    client_dataset = FDS.load_partition(partition_id, "train")
    client_dataset = reformat(client_dataset, llm_task=llm_task)

    ratio = max(0.0, min(eval_split_ratio, 1.0))
    num_examples = len(client_dataset)
    task_key = _normalize_task_name(llm_task)

    def _cache_key(split: str) -> Tuple[int, str, int, float, int, str, str]:
        return (
            partition_id,
            dataset_name,
            num_partitions,
            ratio,
            eval_split_seed,
            task_key,
            split,
        )

    if ratio <= 0.0 or num_examples <= 1:
        PARTITION_CACHE[_cache_key("train")] = client_dataset
        PARTITION_CACHE[_cache_key("eval")] = None
        return

    eval_size = max(1, int(num_examples * ratio))
    if eval_size >= num_examples:
        eval_size = num_examples - 1

    if eval_size <= 0:
        PARTITION_CACHE[_cache_key("train")] = client_dataset
        PARTITION_CACHE[_cache_key("eval")] = None
        return

    split_ds = client_dataset.train_test_split(
        test_size=eval_size,
        seed=eval_split_seed,
        shuffle=True,
    )
    PARTITION_CACHE[_cache_key("train")] = split_ds["train"]
    PARTITION_CACHE[_cache_key("eval")] = split_ds["test"]


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    split: str = "train",
    eval_split_ratio: float = 0.0,
    eval_split_seed: int = 2024,
    llm_task: Optional[str] = None,
) -> Optional[Dataset]:
    """Load partition data.

    Args:
        partition_id: 현재 클라이언트 파티션 ID.
        num_partitions: 총 파티션 수.
        dataset_name: Hugging Face 데이터셋 이름.
        split: ``\"train\"`` 또는 ``\"eval\"`` 중 선택.
        eval_split_ratio: 검증 데이터 비율(0.0~1.0).
        eval_split_seed: 검증 데이터 분할 시드.

    Returns:
        요청된 split에 해당하는 ``datasets.Dataset`` (또는 평가 데이터가 없을 경우 ``None``).
    """
    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported split '{split}'")

    ratio = 0.0 if eval_split_ratio is None else eval_split_ratio
    seed = eval_split_seed
    normalized_task = _normalize_task_name(llm_task or DEFAULT_TASK_NAME)

    cache_key = (
        partition_id,
        dataset_name,
        num_partitions,
        max(0.0, min(ratio, 1.0)),
        seed,
        normalized_task,
        split,
    )
    if cache_key not in PARTITION_CACHE:
        _populate_partition_cache(
            partition_id,
            num_partitions,
            dataset_name,
            ratio,
            seed,
            normalized_task,
        )

    return PARTITION_CACHE.get(cache_key)
