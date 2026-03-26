# QLoRA 训练脚本

本目录用于训练本地 Qwen 4B 基座模型的 QLoRA 适配器。

## 目录

- `train_qlora.py`: 主训练脚本
- `run_train.ps1`: PowerShell 启动脚本
- `requirements.txt`: Python 依赖

## 重要说明

如果你当前“只有 Ollama 本地模型”，要先确认你手里的基座模型是不是 Hugging Face 兼容目录。

- 可以直接训练：`config.json`、`tokenizer.json`、`*.safetensors` 这类 Transformers/HF 格式目录
- 不能直接训练：Ollama 的 `GGUF` 或内部 `blob` 文件

QLoRA 微调的是 Transformers 模型权重，不是 Ollama 打包后的推理文件。

## 安装依赖

```powershell
pip install -r train/requirements.txt
```

## 启动训练

```powershell
.\train\run_train.ps1 -ModelNameOrPath "D:\models\qwen3.5-4b"
```

也可以直接运行：

```powershell
python train/train_qlora.py `
  --model_name_or_path "D:\models\qwen3.5-4b" `
  --train_file "risk_data/train.jsonl" `
  --eval_file "risk_data/val.jsonl" `
  --output_dir "output/qlora-qwen3.5-4b"
```

## 默认输出

训练结果会写到：

```text
output/qlora-qwen3.5-4b
```

包括：

- LoRA adapter
- tokenizer 副本
- checkpoint
- `training_summary.json`
