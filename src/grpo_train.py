import torch
import torch.nn.functional as F
from tqdm import tqdm
from .config import cfg
from .data_loader import prepare_data
from .ppo_train import PPOTrainer 

class GRPOTrainer(PPOTrainer):
    def train_step(self, prompt_ids, attention_mask):
        # GRPO logic: sample G responses for each prompt
        group_size = 4
        
        group_prompts = prompt_ids.repeat_interleave(group_size, dim=0)
        group_mask = attention_mask.repeat_interleave(group_size, dim=0)

        with torch.no_grad():
            responses = self.policy.generate(
                group_prompts, 
                attention_mask=group_mask,
                max_new_tokens=50, 
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_p=0.9
            )
            
            full_mask = (responses != self.tokenizer.pad_token_id).long()
            # score responses
            rewards = self.reward_model(responses, attention_mask=full_mask).logits.squeeze(-1)
            ref_log_probs = self.get_log_probs(self.ref_model, responses, full_mask)

        # compute group advantages
        rewards_grouped = rewards.view(-1, group_size)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        
        # advantage calc
        advantages = (rewards_grouped - mean_rewards) / std_rewards # [B, G]
        advantages = advantages.view(-1)
        advantages = advantages.unsqueeze(-1) 
        # policy update
        new_log_probs = self.get_log_probs(self.policy, responses, full_mask)
        ratio = torch.exp(new_log_probs - ref_log_probs)
        kl = new_log_probs - ref_log_probs
        
        # loss calc
        loss = -(advantages.detach() * ratio - 0.1 * kl).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def run(self):
        print("GRPO training")
        data = prepare_data("train")
        prompts = [x.split("\n\nAssistant:")[0] + "\n\nAssistant:" for x in data["chosen"]]
        
        num_batches = len(prompts) // cfg.batch_size
        print(f"Total Batches: {num_batches}")
        
        for i in tqdm(range(num_batches)):
            batch_prompts = prompts[i*cfg.batch_size : (i+1)*cfg.batch_size]
            
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(cfg.device)
            
            try:
                loss = self.train_step(inputs["input_ids"], inputs["attention_mask"])
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Skipping batch {i} due to OOM")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
        self.policy.save_pretrained(f"{cfg.output_dir}/grpo_model")