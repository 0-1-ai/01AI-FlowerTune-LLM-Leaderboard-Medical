"""mmfl: A Flower / FlowerTune app.

모델 로딩과 양자화 설정(4/8비트), LoRA 적용, 코사인 학습률 스케줄을 제공합니다.
GPU 가용성에 따라 안전하게 동작하도록 CPU 폴백(옵션)을 지원합니다.
"""

import math
import os

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule.

    코사인 함수를 이용해 학습률을 점진적으로 감소시키는 스케줄입니다.
    current_round는 1부터 total_round까지 증가합니다.
    """
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig, force_cpu: bool = False):
    """Load model with appropriate quantization config and other optimizations.

    - 4/8비트 양자화로 VRAM 사용량을 절감
    - k-bit 학습 준비 및 LoRA 어댑터 적용
    - 필요 시 gradient checkpointing 활성화로 메모리 추가 절감
    """
    # 1) 양자화 모드 해석(0=비활성화, 4/8=bitsandbytes)
    quant = getattr(model_cfg, "quantization", 4)

    # force_cpu가 True이면, CPU 로딩을 위해 양자화를 비활성화
    if force_cpu:
        quant = 0

    # 2) GPU 가용성 확인 및 CPU 폴백 정책
    cuda_available = torch.cuda.is_available() and not force_cpu
    allow_cpu_fallback = os.getenv("MMFL_CPU_FALLBACK", "0").lower() in {"1", "true", "yes"}

    # 3) 양자화 구성 결정
    quantization_config = None
    if quant in (4, 8):
        if not cuda_available and not allow_cpu_fallback:
            raise RuntimeError(
                "CUDA 미사용 환경에서 4/8비트 bitsandbytes 양자화를 사용할 수 없음. "
                "옵션: (1) 이 파드에 GPU 할당, (2) 환경변수 MMFL_CPU_FALLBACK=1 설정으로 CPU 로딩(매우 느리고 메모리 소모 큼), "
                "(3) 구성에서 model.quantization=0으로 비양자화 로딩."
            )
        if cuda_available:
            # GPU가 있을 때만 bnb 양자화 적용
            quantization_config = (
                BitsAndBytesConfig(load_in_4bit=True)
                if quant == 4
                else BitsAndBytesConfig(load_in_8bit=True)
            )
        else:
            # CPU 폴백(양자화 비활성화)
            quantization_config = None
            quant = 0
    elif quant in (0, None):
        quantization_config = None
    else:
        raise ValueError(f"quantization은 0/4/8 중 하나여야 함. 전달값: {quant}")

    # 4) 모델 로딩(dtype은 환경에 맞게 선택)
    dtype = torch.bfloat16 if (cuda_available and quant in (4, 8)) or force_cpu else torch.float32
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    # k-bit 학습 준비(옵션으로 gradient checkpointing 사용)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    # LoRA 어댑터 설정: 랭크/알파/드롭아웃 값은 구성에서 전달
    lora_cfg = model_cfg.lora

    target_modules_cfg = getattr(lora_cfg, "target_modules", None)
    if target_modules_cfg is None:
        target_modules_cfg = getattr(lora_cfg, "peft_lora_target_modules", None)

    target_modules_list = None
    if isinstance(target_modules_cfg, str):
        normalized = target_modules_cfg.strip()
        if normalized:
            if normalized.lower() == "all":
                target_modules_list = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                ]
            else:
                target_modules_list = [
                    module.strip()
                    for module in normalized.split(",")
                    if module.strip()
                ]
    elif isinstance(target_modules_cfg, (list, tuple)):
        target_modules_list = [
            str(module).strip()
            for module in target_modules_cfg
            if str(module).strip()
        ]
        if not target_modules_list:
            target_modules_list = None

    model_name_lower = str(model_cfg.name).lower()
    if target_modules_list is None and "qwen" in model_name_lower:
        target_modules_list = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    use_dora = bool(getattr(lora_cfg, "use_dora", False))

    peft_config = LoraConfig(
        r=lora_cfg.peft_lora_r,
        lora_alpha=lora_cfg.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
        target_modules=target_modules_list,
        use_dora=use_dora,
    )

    if model_cfg.gradient_checkpointing:
        # 체크포인팅 시 캐시 사용 비활성화(메모리 절감을 위해)
        model.config.use_cache = False

    return get_peft_model(model, peft_config)
