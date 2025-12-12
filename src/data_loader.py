from datasets import load_dataset
from transformers import AutoTokenizer
from .config import cfg

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    return tokenizer

def prepare_data(split="train"):
    data = load_dataset(cfg.dataset_name, split=split)
    
    # filter to remove hallcucinations / unhelpful answers
    def is_clean(example):
        text = example["chosen"]
        if len(text) < 200: return False
        bad_phrases = ["I'm not sure", "I am not sure", "I cannot", "I can't", "don't know", "sorry", "apologize"]
        if any(p in text for p in bad_phrases): return False
        return True

    print(f"Original: {len(data)}")
    data = data.filter(is_clean)
    print(f"Cleaned: {len(data)}")
    
    if len(data) > 2000:
        data = data.select(range(2000))

    return data

def preprocess_reward_data(examples, tokenizer):
    inputs_chosen = tokenizer(examples["chosen"], truncation=True, max_length=cfg.max_length, padding="max_length")
    inputs_rejected = tokenizer(examples["rejected"], truncation=True, max_length=cfg.max_length, padding="max_length")
    
    return {
        "input_ids_chosen": inputs_chosen["input_ids"],
        "attention_mask_chosen": inputs_chosen["attention_mask"],
        "input_ids_rejected": inputs_rejected["input_ids"],
        "attention_mask_rejected": inputs_rejected["attention_mask"],
    }