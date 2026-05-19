from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Prepare model for training (freeze layers, cast to float32 for stability)
model = prepare_model_for_kbit_training(model)

# 2. LoRA Configuration
# Target modules: q_proj, v_proj are standard. 
# Note: Usually we DO NOT train the vision_tower (mm_projector is optional)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Expected: ~1-2% of parameters trainable.