# Llama3.1-Aloe-Beta-8B - Medical (Run v1)

This directory contains the federated instruction tuning submission for the **Finance** challenge using the [HPAI-BSC/Llama3.1-Aloe-Beta-8B](https://huggingface.co/HPAI-BSC/Llama3.1-Aloe-Beta-8B) model on the [flwrlabs/medical-meadow-medical-flashcards](https://huggingface.co/datasets/flwrlabs/medical-meadow-medical-flashcards) dataset.

We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in a federated way,
allowing users to perform the training on a single GPU.

## Project Structure

```text
.
├── flowertune-medical/              # Source code for ClientApp, ServerApp, and Strategy
├── flowertune-eval-medical/         # Evaluation scripts and instructions
├── pyproject.toml                   # Project configuration and dependencies
└── README.md                        # This file
```

## Methodology

This submission performs federated LLM fine-tuning with **LoRA** using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with the **FedAvg** strategy.

### Model Configuration
- **Base Model**: `HPAI-BSC/Llama3.1-Aloe-Beta-8B`
- **Quantization**: 4-bit
- **PEFT**: LoRA (Rank: 32, Alpha: 128)
- **Target Modules**: `q_proj`, `v_proj`

### Training Configuration
- **Rounds**: 10
- **Fraction Fit**: 0.1 (10% of clients per round)
- **Local Epochs**: 3
- **Optimizer**: Paged AdamW 8-bit

## Prerequisites

Before running the simulation, ensure you have access to the model and are logged into Hugging Face.

1. **Model Access**: Ensure you have access to [InternLM3 8B Instruct](https://huggingface.co/internlm/internlm3-8b-instruct) on Hugging Face.
2. **Hugging Face Login**:
   ```bash
   huggingface-cli login
   ```

## Setup & Running

1. **Install Dependencies**:
   Ensure you are in this directory (`submissions/medical/llama3.1-aloe-8b-v1`).
   ```bash
   pip install -e .
   ```

2. **Run Simulation**:
   Run the challenge with default config values defined in `pyproject.toml`.
   ```bash
   flwr run
   ```

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` in `pyproject.toml` are not allowed to be modified for fair competition if you plan to participate in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

## Experimental Setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of 10 rounds.

## VRAM Consumption & Resources

You can adjust the CPU/GPU resources assigned to each client based on your device capabilities by modifying `options.backend.client-resources.num-cpus` and `options.backend.client-resources.num-gpus` under `[tool.flwr.federations.local-simulation]` entry in `pyproject.toml`.

Experiments were run on RTX 3090/4090 class GPUs with 4-bit quantization.

## Model Saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the server side as default, which can be specified with `train.save-every-round` under `[tool.flwr.app.config]` entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participate in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

## Changes from Baseline
- Base model: switched from `mistralai/Mistral-7B-v0.3` to `HPAI-BSC/Llama3.1-Aloe-Beta-8B` with `trust_remote_code=true`.
- Rounds: reduced from 200 to 10.
- LoRA: rank/alpha `32/128` and target modules `q_proj, v_proj` (baseline: `32/64`, default targets).
- Learning Rate: increased compared to the baseline, from `5e-5 / 1e-6` to `1e-4 / 1e-5` (max / min).
- Training batch: per-device batch size 16 with gradient accumulation 2 (effective batch 32) instead of per-device 16, accumulation 1.
- Torch/runtime stack: `torch==2.4.0`, `peft==0.14.0`, `transformers==4.50.3` (baseline uses `torch==2.9.1`, `peft==0.6.2`).

## Evaluation

See `evaluation/README.md` for the exact environment setup and the single-line command to run. Results are stored under `evaluation/benchmarks/` (acc/generation artifacts already included).

### Results (peft_10)

|         | PubMedQA | MedMCQA | MedQA | CareQA |  Avg  |
| :-----: | :------: | :-----: | :---: | :---:  | :---: |
| FedAvg |   74.80  |  55.39  | 59.31 | 64.79  | 63.57 |

**Communication budget: 2080.93 MB**

## Checkpoints

- Round 10 PEFT adapter: [Google Drive link](https://drive.google.com/file/d/1QLx0u9kWb4_8FuK7xIDuxWblOIUlfWhA/view?usp=sharing)