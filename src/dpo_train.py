import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import cfg
from .data_loader import get_tokenizer, prepare_data, preprocess_reward_data
from .ppo_train import PPOTrainer

class DPOTrainer(PPOTrainer):
    def train_step(self, batch):
        pos_ids = batch['input_ids_chosen'].to(cfg.device)
        pos_mask = batch['attention_mask_chosen'].to(cfg.device)
        
        neg_ids = batch['input_ids_rejected'].to(cfg.device)
        neg_mask = batch['attention_mask_rejected'].to(cfg.device)
        
        # policy log probs 
        policy_pos_log = self.get_log_probs(self.policy, pos_ids, pos_mask).sum(-1)
        policy_neg_log = self.get_log_probs(self.policy, neg_ids, neg_mask).sum(-1)
        
        with torch.no_grad():
            ref_pos_log = self.get_log_probs(self.ref_model, pos_ids, pos_mask).sum(-1)
            ref_neg_log = self.get_log_probs(self.ref_model, neg_ids, neg_mask).sum(-1)
            
        # loss calc
        beta = 0.1
        logits = (policy_pos_log - ref_pos_log) - (policy_neg_log - ref_neg_log)
        
        loss = -torch.nn.functional.logsigmoid(beta * logits).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def run(self):
        print("DPO training")
        dataset = prepare_data("train")
        dataset = dataset.map(lambda x: preprocess_reward_data(x, self.tokenizer), batched=True)
        dataset.set_format(type="torch", columns=["input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"])
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        
        for batch in tqdm(loader):
            loss = self.train_step(batch)

        self.policy.save_pretrained(f"{cfg.output_dir}/dpo_model")