import json

# Input file: your dataset in multi-turn messages format
input_file = "dataset.jsonl"      # replace with your filename
output_file = "dataset1.jsonl"    # output for LoRA training

# Load your multi-turn dataset
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Open output file
with open(output_file, "w", encoding="utf-8") as out_f:
    for sample in data:
        messages = sample.get("messages", [])
        
        # Find last user message
        last_user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg["role"] == "user":
                last_user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]
        
        # Skip if any part missing
        if last_user_msg and assistant_msg:
            # Prepare JSONL line
            jsonl_obj = {
                "prompt": last_user_msg.strip(),
                "completion": assistant_msg.strip()
            }
            out_f.write(json.dumps(jsonl_obj, ensure_ascii=False) + "\n")

print(f"✅ Conversion complete! Output saved to {output_file}")