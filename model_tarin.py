from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1. Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl")

# 2. Load tokenizer
model_name = "meta-llama/Llama-3-8b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Tokenize dataset
def tokenize(example):
    # Add BOS/EOS if needed
    prompt = example["prompt"]
    completion = example["completion"]
    full_text = f"{prompt}\n{completion}"
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# 4. Load base model (CPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",   # Use CPU
    low_cpu_mem_usage=True
)

# 5. Apply LoRA
lora_config = LoraConfig(
    r=8,              # LoRA rank (low for CPU)
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # LLaMA attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 6. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="lora_llama8b",
    num_train_epochs=3,
    per_device_train_batch_size=1,   # CPU-friendly
    gradient_accumulation_steps=4,   # simulate larger batch
    learning_rate=5e-4,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    fp16=False,                      # CPU -> no fp16
    report_to="none",
    warmup_steps=10
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 9. Train
trainer.train()

# 10. Save LoRA weights
model.save_pretrained("lora_llama8b")