import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import sys
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

tokenizer = AutoTokenizer.from_pretrained(
    'Qwen-7B-Chat',
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen-7B-Chat',
    device_map="auto",
    trust_remote_code=True
).eval()

peft_model_id = "lora-checkpoint-4000"
config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(model, peft_model_id)
