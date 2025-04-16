At the time of writing, Together AI doesn't support adding files via API or Python SDK, so we've included instructions instead of a code file to copy.

1. Set your TOGETHER_API_KEY and WANDB_API_KEY
2. Upload your training file: `together files upload combined_datasets_for_tuning/message_format/train.jsonl`
3. Upload your validation file: `together files upload combined_datasets_for_tuning/message_format/validation.jsonl`
4. View file ids: `together files list`
5. Run these CLI commands to start training jobs:
```
together fine-tuning create \
  --training-file "your file id" \
  --validation-file "your validation file id" \
  --model "Qwen/Qwen2-1.5B-Instruct" \
  --lora \
  --n-epochs 3 \
  --n-evals 10 \
  --learning-rate 0.00003 \
  --batch-size 32 \
  --warmup-ratio 0.1 \
  --wandb-api-key $WANDB_API_KEY
```