import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForTokenClassification
from torch.optim import AdamW
from tqdm import tqdm
from .config import cfg
from .data_loader import get_tokenizer, prepare_data

class PPOTrainer:
    def __init__(self):
        self.tokenizer = get_tokenizer()

        # policy model from SFT
        self.policy = AutoModelForCausalLM.from_pretrained(f"{cfg.output_dir}/sft_model").to(cfg.device)
        self.policy.config.pad_token_id = self.tokenizer.pad_token_id 
        
        # 2. reference model - frozen SFT
        self.ref_model = AutoModelForCausalLM.from_pretrained(f"{cfg.output_dir}/sft_model").to(cfg.device)
        self.ref_model.eval()
        self.ref_model.config.pad_token_id = self.tokenizer.pad_token_id 
        
        # 3. reward model (sequence classifier for final score)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(f"{cfg.output_dir}/reward_model").to(cfg.device)
        self.reward_model.eval()
        self.reward_model.config.pad_token_id = self.tokenizer.pad_token_id 
        
        # 4. critic (value model)
        self.critic = AutoModelForTokenClassification.from_pretrained(cfg.model_name, num_labels=1).to(cfg.device)
        self.critic.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.optimizer = AdamW(list(self.policy.parameters()) + list(self.critic.parameters()), lr=cfg.ppo_lr)
        
        # hyperparams
        self.clip_eps = 0.2
        self.kl_coef = 0.1
        self.gamma = 0.99
        self.lam = 0.95

    def get_log_probs(self, model, input_ids, attention_mask=None):
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs

    def train_step(self, prompt_ids, attention_mask):
        with torch.no_grad():
            response_ids = self.policy.generate(
                prompt_ids, 
                attention_mask=attention_mask,
                max_new_tokens=50, 
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True 
            )
            
            # attention mask
            full_mask = (response_ids != self.tokenizer.pad_token_id).long()
            # get reward model score
            rm_score = self.reward_model(response_ids, attention_mask=full_mask).logits.squeeze(-1)
            ref_log_probs = self.get_log_probs(self.ref_model, response_ids, full_mask)
        
        for _ in range(4): 
            new_log_probs = self.get_log_probs(self.policy, response_ids, full_mask)
            
            ratio_log = new_log_probs - ref_log_probs
            kl_penalty = self.kl_coef * ratio_log
            
            rewards = torch.zeros_like(new_log_probs)
            rewards[:, -1] += rm_score 
            rewards -= kl_penalty 
            
            values = self.critic(response_ids, attention_mask=full_mask).logits.squeeze(-1)
            values = values[:, :-1]
            
            # GAE calc
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(rewards.size(1))):
                nextnonterminal = 1.0 if t < rewards.size(1) - 1 else 0.0
                nextvalues = values[:, t+1] if t < rewards.size(1) - 1 else 0.0
                delta = rewards[:, t] + self.gamma * nextvalues * nextnonterminal - values[:, t]
                advantages[:, t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # policy loss
            ratio = torch.exp(new_log_probs - ref_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, rewards + values.detach())
            
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

    def run(self):
        print("PPO Training")
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
            
            if i % 10 == 0:
                pass

        self.policy.save_pretrained(f"{cfg.output_dir}/ppo_model")