from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoTokenizer

# ——— 模型 & 分词器 ———
model_path = "."
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Qwen2.5 没有默认 pad_token，使用 eos_token 充当
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ——— 数据集（本地 JSON/JSONL，每行 {"messages": [{"role":"...", "content":"..."}, ...]}）———
dataset = load_dataset(
    "json",
    data_files="./data/sft_chat_22k.jsonl",
    split="train",
)

# ——— 训练配置 ———
training_args = SFTConfig(
    output_dir="./sft_22k",

    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,

    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,

    max_length=2048,
    bf16=True,

    logging_steps=10,
    save_steps=200,
    save_total_limit=5,
    
    report_to="wandb",
    run_name="sft_22k",
)

# ——— Trainer ———
trainer = SFTTrainer(
    model=model_path,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()