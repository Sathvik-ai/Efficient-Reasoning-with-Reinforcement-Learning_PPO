import os, torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import torch.nn.functional as F
from typing import List

# 1. Config & Setup
MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
MAX_SEQ_LENGTH = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RL_CONFIG = {'n_steps': 100, 'batch_size': 4, 'n_samples': 8, 'ppo_clip': 0.2, 'kl_beta': 0.02, 'lr': 1e-5}

# 2. Dataset
def load_math_benchmark(benchmark_name="GSM8K", limit=300):
    print(f"Loading {benchmark_name}...")
    conversations = []
    ds = load_dataset("gsm8k", "main", split="train").select(range(limit))
    for item in ds:
        conversations.append({
            "question": item["question"],
            "final_answer": item["answer"].split("####")[-1].strip(),
            "answer": item["answer"]
        })
    return conversations

conversations = load_math_benchmark(limit=300)

# 3. Model & Tokenizer Initialization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
ref_model, _ = FastLanguageModel.from_pretrained(model_name=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)

# 4. RLOO Advantage
class RLOOAdvantage:
    @staticmethod
    def compute(rewards: List[float]) -> torch.Tensor:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        n = len(rewards_tensor)
        advantages = []
        for i in range(n):
            others = torch.cat([rewards_tensor[:i], rewards_tensor[i+1:]])
            advantages.append(rewards_tensor[i] - others.mean())
        advantages = torch.stack(advantages)
        std = advantages.std()
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        return advantages

# 5. Reward Function
def compute_reward(output: str, gt: str) -> float:
    # Simplified exact match + length penalty
    correct = 1.0 if gt in output else 0.0
    length_penalty = -0.01 * len(output.split())
    return correct + length_penalty

# 6. RLOO Trainer Step
def rloo_step(model, ref_model, tokenizer, prompt, gt, optimizer):
    model.train()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    generated_texts, log_probs_list, rewards = [], [], []
    for _ in range(RL_CONFIG['n_samples']):
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(gen_text)
        
        # Log probs
        with torch.no_grad():
            logits = model(outputs).logits
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_list.append(log_probs.mean())
        
        # Reward
        rewards.append(compute_reward(gen_text, gt))
        
    advantages = RLOOAdvantage.compute(rewards).to(DEVICE)
    log_probs_tensor = torch.stack(log_probs_list)
    loss = -(advantages * log_probs_tensor).mean()
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# 7. Training Loop Implementation
print("Starting RLOO Training Loop...")
# optimizer = torch.optim.AdamW(model.parameters(), lr=RL_CONFIG['lr'])
# for step, conv in enumerate(conversations[:RL_CONFIG['n_steps']]):
#     loss = rloo_step(model, ref_model, tokenizer, conv['question'], conv['final_answer'], optimizer)
#     print(f"Step {step} | Loss: {loss:.4f}")
