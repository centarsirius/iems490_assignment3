import torch
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "gpt2"

    dataset_name: str = "Anthropic/hh-rlhf"
    output_dir: str = "./output_final"

    seed: int = 42
    max_length: int = 256       
    batch_size: int = 2          
    grad_accumulation: int = 4  
    
    # learning rates 
    sft_lr: float = 1e-4         
    rm_lr: float = 2e-5
    ppo_lr: float = 1e-6         
    
    # test/train
    debug: bool = False          
    train_size: int = 3000      

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()