"""Main training entry point for Efficient Reasoning with PPO."""

import argparse
import dataclasses
import json
import logging
import os

import torch

from config import DataConfig, ModelConfig, PPOConfig, RewardConfig, TrainingConfig
from data_utils import get_dataset
from model_utils import count_parameters, get_model_and_tokenizer, save_model
from ppo_trainer import PPOTrainer
from reward_functions import build_reward_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a language model for efficient reasoning with PPO."
    )
    # Model
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_lora", action="store_true", default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    # Data
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--max_response_length", type=int, default=None)
    parser.add_argument("--num_train_samples", type=int, default=None)
    # PPO
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--rollout_batch_size", type=int, default=None)
    parser.add_argument("--mini_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    # Reward
    parser.add_argument("--reward_type", type=str, default=None)
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file (overridden by CLI args).",
    )
    # Misc
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def merge_config_with_args(cfg: TrainingConfig, args: argparse.Namespace) -> None:
    """Override config fields with non-None CLI arguments."""
    field_map = {
        "model_name_or_path": ("model", "model_name_or_path"),
        "use_lora": ("model", "use_lora"),
        "lora_r": ("model", "lora_r"),
        "dataset_name": ("data", "dataset_name"),
        "max_prompt_length": ("data", "max_prompt_length"),
        "max_response_length": ("data", "max_response_length"),
        "num_train_samples": ("data", "num_train_samples"),
        "num_train_epochs": ("ppo", "num_train_epochs"),
        "rollout_batch_size": ("ppo", "rollout_batch_size"),
        "mini_batch_size": ("ppo", "mini_batch_size"),
        "learning_rate": ("ppo", "learning_rate"),
        "output_dir": ("ppo", "output_dir"),
        "reward_type": ("reward", "reward_type"),
        "seed": ("ppo", "seed"),
    }
    for arg_name, (sub_cfg_name, field_name) in field_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(getattr(cfg, sub_cfg_name), field_name, value)

    if args.device is not None:
        cfg.device = args.device


def main() -> None:
    args = parse_args()

    # Load base config from file if provided
    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        cfg = TrainingConfig(**cfg_dict)
    else:
        cfg = TrainingConfig()

    # Override with CLI args
    merge_config_with_args(cfg, args)

    logger.info("Effective configuration:\n%s", json.dumps(dataclasses.asdict(cfg), indent=2))

    # Seed
    torch.manual_seed(cfg.ppo.seed)

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading model '%s'…", cfg.model.model_name_or_path)
    policy, ref_model, tokenizer = get_model_and_tokenizer(
        cfg.model, device=cfg.device, load_reference=True
    )
    trainable, total = count_parameters(policy)
    logger.info(
        "Policy parameters: %d trainable / %d total (%.1f%%)",
        trainable, total, 100 * trainable / total,
    )

    # ------------------------------------------------------------------
    # 2. Load datasets
    # ------------------------------------------------------------------
    logger.info("Loading dataset '%s'…", cfg.data.dataset_name)
    train_examples, eval_examples = get_dataset(cfg.data)
    logger.info(
        "Dataset loaded: %d train | %d eval",
        len(train_examples), len(eval_examples),
    )

    # ------------------------------------------------------------------
    # 3. Build reward function
    # ------------------------------------------------------------------
    reward_fn = build_reward_function(cfg.reward)
    logger.info("Reward function: %s", cfg.reward.reward_type)

    # ------------------------------------------------------------------
    # 4. Create trainer and start training
    # ------------------------------------------------------------------
    trainer = PPOTrainer(
        policy=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        cfg=cfg.ppo,
        reward_cfg=cfg.reward,
    )

    logger.info("Starting PPO training…")
    trainer.train(
        train_examples=train_examples,
        eval_examples=eval_examples,
        max_prompt_length=cfg.data.max_prompt_length,
        max_response_length=cfg.data.max_response_length,
    )

    # ------------------------------------------------------------------
    # 5. Save final model
    # ------------------------------------------------------------------
    final_dir = os.path.join(cfg.ppo.output_dir, "final")
    save_model(policy, tokenizer, final_dir)
    logger.info("Training complete. Final model saved to '%s'.", final_dir)


if __name__ == "__main__":
    main()
