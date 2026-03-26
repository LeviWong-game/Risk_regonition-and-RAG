param(
    [string]$ModelNameOrPath = "",
    [string]$TrainFile = "risk_data/train.jsonl",
    [string]$EvalFile = "risk_data/val.jsonl",
    [string]$OutputDir = "output/qlora-qwen3.5-4b"
)

if (-not $ModelNameOrPath) {
    Write-Error "请传入 -ModelNameOrPath。它必须是一个 Hugging Face 兼容的本地模型目录。Ollama 的 GGUF/blob 不能直接用 QLoRA 微调。"
    exit 1
}

python train/train_qlora.py `
  --model_name_or_path $ModelNameOrPath `
  --train_file $TrainFile `
  --eval_file $EvalFile `
  --output_dir $OutputDir
