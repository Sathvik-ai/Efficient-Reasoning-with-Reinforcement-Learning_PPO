# Efficient Reasoning with Reinforcement Learning (PPO)

Train language models to reason more efficiently using **Proximal Policy Optimization (PPO)**.  
The framework supports rule-based and reward-model-based reward signals, LoRA for
parameter-efficient fine-tuning, and is tested end-to-end on math reasoning benchmarks
(GSM8K, MATH).

---

## Project Structure

```
.
├── config.py            # Dataclass-based configuration (model, data, PPO, reward)
├── data_utils.py        # Dataset loading, prompt building, batching helpers
├── model_utils.py       # Model / tokenizer loading, LoRA wrapping, checkpoint saving
├── ppo_trainer.py       # Core PPO training loop (rollout → GAE → policy update)
├── reward_functions.py  # Rule-based and reward-model reward functions
├── train.py             # CLI entry point
├── tests.py             # Unit + integration tests for all components
└── requirements.txt     # Python dependencies
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training (GSM8K, default settings)

```bash
python train.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name openai/gsm8k \
  --output_dir outputs/ppo_gsm8k \
  --num_train_epochs 1 \
  --rollout_batch_size 32 \
  --learning_rate 1e-6
```

### 3. Use a JSON config file

```bash
python train.py --config my_config.json
```

CLI arguments override values in the JSON file.

---

## Key Components

### `config.py`

Four nested dataclasses:

| Class | Purpose |
|---|---|
| `ModelConfig` | Base model path, LoRA settings, dtype |
| `DataConfig` | Dataset name/split, prompt length, system prompt |
| `PPOConfig` | Clip epsilon, KL penalty, optimizer, generation settings |
| `RewardConfig` | Reward type, correctness bonus, normalization |

### `reward_functions.py`

- **`RuleBasedRewardFunction`** – extracts the final answer from the response and
  compares it to the ground truth. Bonus for visible chain-of-thought.
- **`RewardModelFunction`** – uses a fine-tuned sequence-classification model.
- **`build_reward_function(cfg)`** – factory that returns the right function for the config.
- **`normalize_rewards`** – zero-mean unit-variance normalization with clipping.

### `ppo_trainer.py`

- **`PPOTrainer.rollout`** – generates responses, computes rewards, old/reference log-probs,
  values, GAE advantages, and returns.
- **`PPOTrainer._ppo_step`** – clipped surrogate objective + value loss + KL penalty.
- **`AdaptiveKLController`** – adjusts the KL coefficient to track a target KL.
- **`compute_advantages_and_returns`** – GAE (γ, λ) over the token sequence.

### `data_utils.py`

- Loaders for **GSM8K** and **MATH** (Hendrycks et al.).
- `build_chat_messages` / `collate_messages` – applies the tokenizer's chat template.
- `decode_responses` – strips the prompt tokens from the output IDs before decoding.

### `model_utils.py`

- `load_model` / `load_tokenizer` – handles dtype, device map, flash-attention.
- `load_reference_model` – frozen copy of the SFT checkpoint for KL computation.
- `save_model` – optionally merges LoRA adapters before saving.

---

## Running Tests

```bash
python -m pytest tests.py -v
```

or

```bash
python tests.py
```

The test suite covers all components and includes an end-to-end wiring test that
verifies the config → reward → data pipeline without requiring a GPU.

---

## Citation

If you use this codebase, please cite the original PPO paper:

```bibtex
@article{schulman2017ppo,
  title   = {Proximal Policy Optimization Algorithms},
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla
             and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year    = {2017}
}
```
