import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from .config import cfg
from .data_loader import get_tokenizer, prepare_data, preprocess_reward_data

class RewardTrainer:
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=1
        ).to(cfg.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.rm_lr)

    def compute_loss(self, batch):
        # FP - chosen
        chosen_ids = batch["input_ids_chosen"].to(cfg.device)
        chosen_mask = batch["attention_mask_chosen"].to(cfg.device)
        rewards_chosen = self.model(input_ids=chosen_ids, attention_mask=chosen_mask).logits

        # FP - rejected
        rejected_ids = batch["input_ids_rejected"].to(cfg.device)
        rejected_mask = batch["attention_mask_rejected"].to(cfg.device)
        rewards_rejected = self.model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

        # pairwise raking loss: -log(sigmoid(R_chosen - R_rejected))
        loss = -torch.log(torch.sigmoid(rewards_chosen - rewards_rejected)).mean()
        
        # acc tracking
        accuracy = (rewards_chosen > rewards_rejected).float().mean()
        return loss, accuracy

    def train(self):
        dataset = prepare_data("train")
        dataset = dataset.map(lambda x: preprocess_reward_data(x, self.tokenizer), batched=True)
        dataset.set_format(type="torch", columns=["input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"])
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        self.model.train()
        wandb.init(project="rlhf-assignment", name="reward-model-train")

        for epoch in range(1): 
            for step, batch in enumerate(tqdm(dataloader)):
                self.optimizer.zero_grad()
                loss, acc = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    wandb.log({"rm_loss": loss.item(), "rm_acc": acc.item()})

        self.model.save_pretrained(f"{cfg.output_dir}/reward_model")

if __name__ == "__main__":
    trainer = RewardTrainer()
    trainer.train()