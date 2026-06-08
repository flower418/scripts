import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "."
DATA_PATH = "./data/sft_chat_22k.jsonl"
OUTPUT_DIR = "./sft_22k"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,
    max_length=1024,
    eos_token="<|im_end|>",
    save_steps=200,
    save_total_limit=5,
    report_to="wandb",
    run_name="sft_22k",
    model_init_kwargs={"dtype": torch.bfloat16},
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)

trainer = SFTTrainer(
    model=MODEL_PATH,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()
trainer.train()
