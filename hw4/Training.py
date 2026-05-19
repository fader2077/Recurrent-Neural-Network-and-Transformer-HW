from datasets import load_dataset

# Load a tiny demo dataset (e.g., from HuggingFace)
# Students should implement a proper formatting function here
dataset = load_dataset("HuggingFaceM4/ChartQA", split="train[:100]")

def format_data(sample):
    # LLaVA Format: USER: <image>\n<prompt>\nASSISTANT: <answer>
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{sample['query']}"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample['label']}],
        },
    ]
    # Apply processor
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    
    # Process inputs
    inputs = processor(
        text=text_prompt, 
        images=sample['image'], 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    # Labels are input_ids but with padding token masked (-100)
    # (Simplified for demo: usually handled by TRL's SFTTrainer)
    return inputs

# --- Training Setup ---
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./llava-finetuned",
    per_device_train_batch_size=4, # 4090 can handle 4-8 depending on image resolution
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="no"
)

# For HW, students can use the standard Trainer if they handle data collation manually,
# or use SFTTrainer from TRL which is easier for chat formats.