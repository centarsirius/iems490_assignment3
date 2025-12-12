# Assignment 3: Reinforcement Learning from Human Feedback (RLHF)

RLHF pipeline implemented from scratch using PyTorch and Hugging Face Transformers. The main aspect of this project is to align a small language model (`gpt2`, 124M parameters) to human preferences using the Anthropic HH-RLHF dataset.

I compared three alignment algorithms:
1.  **PPO (Proximal Policy Optimization)**: The standard RLHF approach with an explicit Reward Model and Critic.
2.  **GRPO (Group Relative Policy Optimization)**: A critic-free optimization method using group-relative advantages.
3.  **DPO (Direct Preference Optimization)**: An implicit reward method optimizing the policy directly on preference pairs.

## Setup & Requirements
Prerequisites - Docker with NVIDIA Container Toolkit support, GPU with at least 8GB VRAM (Tested on: local NVIDIA TITAN V).

Installation
1. Build and run the Docker container:
```Bash
docker build -t rlhf_env .
docker run --gpus all -it -v $(pwd):/app rlhf_env
```
## Usage (modular)
Reward Model Training (required for PPO/GRPO):
```Bash
python -m src.reward_model
```
Supervised Fine-Tuning (SFT) (initializes policy):

```Bash
python -c "from src.sft_train import train_sft; train_sft()"
```
Policy Optimization:

```Bash
# PPO
python -c "from src.ppo_train import PPOTrainer; PPOTrainer().run()"

# GRPO
python -c "from src.grpo_train import GRPOTrainer; GRPOTrainer().run()"

# DPO
python -c "from src.dpo_train import DPOTrainer; DPOTrainer().run()"
```
Evaluation: Generates evaluation_final.csv comparing Base, SFT, PPO, GRPO, and DPO.

```Bash
python -m src.evaluate
```

## Implementation Details
- Low-Cost Strategy: To accommodate the 124M parameter model constraint, I implemented Layer Freezing during SFT (freezing first 10 layers) to prevent catastrophic forgetting of English syntax as I often ran into the "mode collapse" issue.
- Data Filtering: I filtered the Anthropic HH-RLHF dataset to remove short/low-quality refusals, ensuring the model had sufficient context for alignment.
