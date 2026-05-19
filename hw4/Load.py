# pip install torch transformers accelerate bitsandbytes peft datasets trl pillow
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

model_id = "llava-hf/llava-1.5-7b-hf"

# 1. Quantization Config (Critical for 24GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load Model
print("Loading LLaVA in 4-bit...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Load Processor (Handles Image + Text tokenization)
processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded.")