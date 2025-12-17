# Medical challenge Evaluation Guide (Llama3.1-Aloe-Beta-8B)

We evaluate on Acc ([PubMedQA](https://huggingface.co/datasets/bigbio/pubmed_qa), [MedMCQA](https://huggingface.co/datasets/medmcqa), [MedQA](https://huggingface.co/datasets/bigbio/med_qa), [CareQA](https://huggingface.co/datasets/HPAI-BSC/CareQA)) following the Flower leaderboard rules.

## Environment setup
- Install deps: `pip install -r requirements.txt`
- Hugging Face auth: `huggingface-cli login`

## Run example
- Default: 4bit quantization is required for leaderboard (`--quantization=4`), batch size 16.
- Base model: `HPAI-BSC/Llama3.1-Aloe-Beta-8B`

```bash
python eval.py \
--base-model-name-path=HPAI-BSC/Llama3.1-Aloe-Beta-8B \
--peft-path=./workspace/results/<timestamp>/peft_10 \
--run-name=eval_aloe-beta  \   
--batch-size=16 \
--quantization=4 \
--datasets=pubmedqa,medmcqa,medqa,careqa
```

## Outputs
- Generations/accuracy: `benchmarks/generation_{dataset}_{category}_{run_name}.jsonl`, `benchmarks/acc_{dataset}_{category}_{run_name}.txt`
- No extra public benchmarks beyond `llama3.1-aloe-8b-v1/flowertune-eval-medical/benchmarks`
