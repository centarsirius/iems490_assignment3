import torch
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import cfg
from .data_loader import get_tokenizer, prepare_data

def train_sft():
    print("SFT Training")
    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)

    for name, param in model.named_parameters():
        if "h.10" in name or "h.11" in name or "ln_f" in name:
            param.requires_grad = True 
        else:
            param.requires_grad = False 

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) 
    
    dataset = prepare_data("train")
    
    def preprocess_sft(examples):
        return tokenizer(examples["chosen"], truncation=True, max_length=cfg.max_length, padding="max_length")
    
    dataset = dataset.map(preprocess_sft, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    model.train()
    
    for step, batch in enumerate(tqdm(loader)):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / cfg.grad_accumulation
        loss.backward()
        
        if (step + 1) % cfg.grad_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    model.save_pretrained(f"{cfg.output_dir}/sft_model")  

    print("validating SFT")
    model.eval()
    test_prompt = "Human: What is the capital of France? Assistant:"
    inp = tokenizer(test_prompt, return_tensors="pt").to(cfg.device)
    
    out = model.generate(
        **inp, 
        max_new_tokens=30, 
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    print(f"Sample Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    train_sft()