"""Configuration for Efficient Reasoning with PPO."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the language model."""

    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    # LoRA settings (set use_lora=True to enable parameter-efficient fine-tuning)
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "up_proj", "down_proj", "gate_proj"]
    )


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    eval_dataset_split: str = "test"
    max_prompt_length: int = 512
    max_response_length: int = 1024
    num_train_samples: Optional[int] = None
    num_eval_samples: Optional[int] = 500
    system_prompt: str = (
        "You are a helpful assistant. Think step-by-step before giving your final answer."
    )


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO algorithm."""

    # Training
    num_train_epochs: int = 1
    num_ppo_epochs: int = 1          # PPO update epochs per rollout batch
    steps_per_epoch: int = 1000
    rollout_batch_size: int = 64     # number of prompts per rollout
    mini_batch_size: int = 8         # PPO mini-batch size for gradient updates
    gradient_accumulation_steps: int = 4

    # PPO clipping and objectives
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.0
    kl_penalty: str = "kl"           # "kl" | "abs" | "mse" | "full"
    init_kl_coef: float = 0.1
    target_kl: float = 0.1
    kl_horizon: int = 10_000
    use_adaptive_kl: bool = True

    # Optimizer
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    do_sample: bool = True

    # GAE
    gamma: float = 1.0
    lam: float = 0.95

    # Misc
    seed: int = 42
    log_with: str = "tensorboard"    # "tensorboard" | "wandb" | "none"
    output_dir: str = "outputs/ppo_reasoning"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    reward_type: str = "rule_based"  # "rule_based" | "reward_model"
    # Rule-based rewards
    correct_answer_reward: float = 1.0
    wrong_answer_penalty: float = -0.5
    format_reward: float = 0.1
    length_penalty_coef: float = 0.0
    # Reward model (used when reward_type == "reward_model")
    reward_model_name: str = ""
    reward_model_batch_size: int = 8
    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 5.0


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Hardware
    device: str = "cuda"
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.ppo, dict):
            self.ppo = PPOConfig(**self.ppo)
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)
