from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)
from trl import SFTTrainer


SYSTEM_PROMPT = "你是一个基层矛盾纠纷风险研判助手，负责输出风险等级、风险分值和处置建议。"


@dataclass
class JsonlExample:
    instruction: str
    input: str
    output: str


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        peft_path = checkpoint_path / "adapter_model"
        model = kwargs["model"]
        model.save_pretrained(peft_path)
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for a local Qwen 4B model.")
    parser.add_argument(
        "--model_name_or_path",
        default=os.environ.get("MODEL_NAME_OR_PATH", ""),
        help=(
            "HF-compatible local model directory or model id. "
            "If you only have an Ollama GGUF/blob, export or obtain the corresponding Hugging Face format first."
        ),
    )
    parser.add_argument(
        "--train_file",
        default=str(Path("risk_data") / "train.jsonl"),
        help="Path to the training jsonl.",
    )
    parser.add_argument(
        "--eval_file",
        default=str(Path("risk_data") / "val.jsonl"),
        help="Path to the validation jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path("output") / "qlora-qwen3.5-4b"),
        help="Directory used to store checkpoints, adapters and tokenizer files.",
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules for Qwen-style attention/MLP layers.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[JsonlExample]:
    records: List[JsonlExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                JsonlExample(
                    instruction=payload.get("instruction", "").strip(),
                    input=payload.get("input", "").strip(),
                    output=payload.get("output", "").strip(),
                )
            )
    return records


def build_prompt(example: JsonlExample) -> str:
    user_message = example.instruction
    if example.input:
        user_message = f"{user_message}\n\n{example.input}"
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_message}\n"
        f"<|assistant|>\n{example.output}"
    )


def to_dataset(records: List[JsonlExample]) -> Dataset:
    rows = [{"text": build_prompt(record)} for record in records]
    return Dataset.from_list(rows)


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name_or_path: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model


def trainer_args(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
    )


def main() -> int:
    args = parse_args()
    if not args.model_name_or_path:
        raise ValueError(
            "Missing --model_name_or_path. Provide a local HF-compatible Qwen model directory. "
            "Ollama GGUF blobs cannot be fine-tuned directly with QLoRA."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(args.train_file)
    eval_records = load_jsonl(args.eval_file)
    train_dataset = to_dataset(train_records)
    eval_dataset = to_dataset(eval_records)

    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_model(args.model_name_or_path)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    training_args = trainer_args(args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.add_callback(SavePeftModelCallback())

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "output_dir": args.output_dir,
        "num_train_examples": len(train_records),
        "num_eval_examples": len(eval_records),
        "max_seq_length": args.max_seq_length,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
