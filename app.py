from transformers import pipeline
from peft import PeftModel

model_name = "meta-llama/Llama-3-8b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
model = PeftModel.from_pretrained(base_model, "lora_llama8b")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Show 4 pumps with run status and flow", max_new_tokens=256)
print(output[0]['generated_text'])