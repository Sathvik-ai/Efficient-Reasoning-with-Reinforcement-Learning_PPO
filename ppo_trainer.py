"""Core PPO trainer for efficient LLM reasoning."""

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PPOConfig, RewardConfig
from data_utils import decode_responses
from reward_functions import normalize_rewards

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: log-probabilities
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities for the *response* tokens only.

    Args:
        model: Policy or reference model.
        input_ids: Full sequence (prompt + response), shape ``(B, L)``.
        attention_mask: Attention mask for ``input_ids``.
        response_mask: Boolean mask that is True for response positions.

    Returns:
        Tensor of shape ``(B, L-1)`` with per-token log-probs masked to
        response positions (non-response positions are 0).
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    logits = outputs.logits  # (B, L, V)

    # Shift so that position i predicts token i+1
    shift_logits = logits[:, :-1, :]          # (B, L-1, V)
    shift_labels = input_ids[:, 1:]           # (B, L-1)
    shift_mask = response_mask[:, 1:]         # (B, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                             # (B, L-1)

    return token_log_probs * shift_mask


# ---------------------------------------------------------------------------
# Utility: GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------

def compute_advantages_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and discounted returns.

    For single-step rewards (reward only at the last token), this simplifies
    to a constant advantage equal to the reward minus the mean baseline.

    Args:
        rewards: Shape ``(B, L)`` – non-zero only at the last response token.
        values: Shape ``(B, L)`` – value function estimates.
        response_mask: Boolean mask of shape ``(B, L)`` for response positions.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        advantages, returns – both of shape ``(B, L)``.
    """
    B, L = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(L)):
        mask_t = response_mask[:, t]
        next_value = values[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae * mask_t
        returns[:, t] = (advantages[:, t] + values[:, t]) * mask_t

    # Normalize advantages over the batch
    adv_flat = advantages[response_mask.bool()]
    if adv_flat.numel() > 1:
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    return advantages, returns


# ---------------------------------------------------------------------------
# Adaptive KL controller
# ---------------------------------------------------------------------------

class AdaptiveKLController:
    """Adapts the KL penalty coefficient to track a target KL divergence."""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int) -> None:
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        """Update the coefficient based on the observed KL divergence."""
        proportional_error = (current_kl / self.target) - 1
        adjustment = 1 + proportional_error * (n_steps / self.horizon)
        self.value = max(self.value * adjustment, 1e-6)


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Trains a language model policy using the PPO algorithm.

    Usage::

        trainer = PPOTrainer(policy, ref_model, tokenizer, reward_fn, cfg)
        trainer.train(train_examples, eval_examples)
    """

    def __init__(
        self,
        policy: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_fn: Callable,
        cfg: PPOConfig,
        reward_cfg: Optional[RewardConfig] = None,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.reward_cfg = reward_cfg

        # Value head: a linear layer on top of the last hidden state
        hidden_size = policy.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1, bias=False).to(
            next(policy.parameters()).device
        )

        # Optimizer covers both policy and value head
        self.optimizer = AdamW(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.kl_ctrl = AdaptiveKLController(
            cfg.init_kl_coef, cfg.target_kl, cfg.kl_horizon
        )

        self.global_step = 0
        os.makedirs(cfg.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_responses(
        self,
        batch: Dict[str, Any],
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate responses for a batch of prompts.

        Returns:
            output_ids: Full (prompt + response) token ids, shape ``(B, L)``.
            prompt_lengths: Number of prompt tokens per example, shape ``(B,)``.
        """
        self.policy.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lengths = attention_mask.sum(dim=-1)  # (B,)

        output_ids = self.policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return output_ids, prompt_lengths

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        examples: List[Dict[str, Any]],
        max_prompt_length: int,
        max_response_length: int,
    ) -> Dict[str, Any]:
        """Collect a batch of rollouts.

        Returns a dict with keys:
          input_ids, attention_mask, response_mask, rewards, advantages,
          old_log_probs, ref_log_probs, values.
        """
        from data_utils import collate_messages

        device = next(self.policy.parameters()).device

        # Tokenize prompts using the chat template
        batch = collate_messages(
            examples,
            self.tokenizer,
            max_length=max_prompt_length,
            device=str(device),
        )

        # Generate responses
        output_ids, prompt_lengths = self.generate_responses(batch, max_response_length)

        # Decode responses (new tokens only) for reward computation
        decoded = decode_responses(output_ids, batch["input_ids"], self.tokenizer)
        ground_truths = [ex["answer"] for ex in examples]
        prompts = [ex["prompt"] for ex in examples]

        # Compute scalar rewards
        scalar_rewards = self.reward_fn(decoded, ground_truths, prompts)
        if self.reward_cfg and self.reward_cfg.normalize_rewards:
            scalar_rewards = normalize_rewards(
                scalar_rewards, self.reward_cfg.reward_clip
            )

        B, L = output_ids.shape

        # Build attention mask for full sequence
        full_attention_mask = (output_ids != self.tokenizer.pad_token_id).long()

        # Build response mask (True for generated tokens)
        response_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i, plen in enumerate(prompt_lengths):
            response_mask[i, plen:] = True

        # Place scalar reward at the last response token of each sequence
        rewards_tensor = torch.zeros(B, L, device=device)
        for i in range(B):
            last_pos = (response_mask[i].nonzero(as_tuple=False))
            if last_pos.numel() > 0:
                rewards_tensor[i, last_pos[-1]] = scalar_rewards[i]

        # Compute old log-probs (policy, no grad needed for rollout storage)
        self.policy.eval()
        with torch.no_grad():
            policy_outputs = self.policy(
                input_ids=output_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
            )
        old_log_probs = self._token_log_probs_from_logits(
            policy_outputs.logits, output_ids, response_mask
        )

        # Reference log-probs for KL
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=output_ids,
                attention_mask=full_attention_mask,
            )
        ref_log_probs = self._token_log_probs_from_logits(
            ref_outputs.logits, output_ids, response_mask
        )

        # Values from value head
        hidden_states = policy_outputs.hidden_states[-1]  # (B, L, H)
        values = self.value_head(hidden_states).squeeze(-1)  # (B, L)
        values = values * response_mask

        advantages, returns = compute_advantages_and_returns(
            rewards_tensor, values.detach(), response_mask,
            self.cfg.gamma, self.cfg.lam,
        )

        return {
            "input_ids": output_ids,
            "attention_mask": full_attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_tensor,
            "scalar_rewards": scalar_rewards,
            "advantages": advantages,
            "returns": returns,
            "old_log_probs": old_log_probs.detach(),
            "ref_log_probs": ref_log_probs.detach(),
            "values": values.detach(),
        }

    # ------------------------------------------------------------------
    # PPO update step
    # ------------------------------------------------------------------

    def _ppo_step(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        """Run one PPO update on the stored rollout batch."""
        self.policy.train()

        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]
        response_mask = rollout["response_mask"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        old_log_probs = rollout["old_log_probs"]
        ref_log_probs = rollout["ref_log_probs"]

        # Forward pass
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        new_log_probs = self._token_log_probs_from_logits(
            outputs.logits, input_ids, response_mask
        )
        hidden_states = outputs.hidden_states[-1]
        new_values = self.value_head(hidden_states).squeeze(-1) * response_mask

        # Policy loss (clipped surrogate objective)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio.clamp(-10, 10))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * response_mask).sum() / (response_mask.sum() + 1e-8)

        # Value loss (clipped)
        value_pred_clipped = rollout["values"] + torch.clamp(
            new_values - rollout["values"], -self.cfg.clip_epsilon, self.cfg.clip_epsilon
        )
        value_loss1 = F.mse_loss(new_values, returns, reduction="none")
        value_loss2 = F.mse_loss(value_pred_clipped, returns, reduction="none")
        value_loss = torch.max(value_loss1, value_loss2)
        value_loss = (value_loss * response_mask).sum() / (response_mask.sum() + 1e-8)

        # KL penalty (approximation)
        kl = (old_log_probs - ref_log_probs) * response_mask
        mean_kl = kl.sum() / (response_mask.sum() + 1e-8)
        kl_loss = self.kl_ctrl.value * mean_kl

        # Total loss
        total_loss = (
            policy_loss
            + self.cfg.value_loss_coef * value_loss
            + kl_loss
        )

        # Entropy bonus
        if self.cfg.entropy_coef > 0:
            shift_logits = outputs.logits[:, :-1, :]
            probs = F.softmax(shift_logits, dim=-1)
            entropy = -(probs * probs.log().clamp(min=-1e9)).sum(-1)
            entropy = (entropy * response_mask[:, 1:]).sum() / (response_mask.sum() + 1e-8)
            total_loss = total_loss - self.cfg.entropy_coef * entropy

        # Gradient update
        total_loss.backward()

        if (self.global_step + 1) % self.cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Update adaptive KL
        if self.cfg.use_adaptive_kl:
            self.kl_ctrl.update(mean_kl.item(), n_steps=1)

        return {
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/kl": mean_kl.item(),
            "loss/total": total_loss.item(),
            "kl_coef": self.kl_ctrl.value,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_examples: List[Dict[str, Any]],
        eval_examples: Optional[List[Dict[str, Any]]],
        max_prompt_length: int,
        max_response_length: int,
    ) -> None:
        """Full PPO training loop."""
        import random

        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.optimizer.zero_grad()

        for epoch in range(self.cfg.num_train_epochs):
            random.shuffle(train_examples)
            batches = [
                train_examples[i: i + self.cfg.rollout_batch_size]
                for i in range(0, len(train_examples), self.cfg.rollout_batch_size)
            ]

            for batch_examples in batches:
                # --- Rollout ---
                rollout = self.rollout(
                    batch_examples, max_prompt_length, max_response_length
                )
                mean_reward = sum(rollout["scalar_rewards"]) / len(rollout["scalar_rewards"])
                logger.info(
                    "Step %d | mean reward: %.4f", self.global_step, mean_reward
                )

                # --- PPO epochs ---
                for ppo_epoch in range(self.cfg.num_ppo_epochs):
                    metrics = self._ppo_step(rollout)
                    self.global_step += 1

                    if self.global_step % self.cfg.logging_steps == 0:
                        logger.info(
                            "Step %d | %s | reward: %.4f",
                            self.global_step,
                            " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()),
                            mean_reward,
                        )

                    if (
                        eval_examples is not None
                        and self.global_step % self.cfg.eval_steps == 0
                    ):
                        acc = self.evaluate(
                            eval_examples, max_prompt_length, max_response_length
                        )
                        logger.info(
                            "Step %d | eval accuracy: %.4f", self.global_step, acc
                        )

                    if self.global_step % self.cfg.save_steps == 0:
                        ckpt_dir = os.path.join(
                            self.cfg.output_dir, f"checkpoint-{self.global_step}"
                        )
                        from model_utils import save_model

                        save_model(self.policy, self.tokenizer, ckpt_dir)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        eval_examples: List[Dict[str, Any]],
        max_prompt_length: int,
        max_response_length: int,
        batch_size: int = 8,
    ) -> float:
        """Evaluate the policy and return accuracy on eval_examples."""
        from data_utils import collate_messages
        from reward_functions import is_correct

        self.policy.eval()
        device = next(self.policy.parameters()).device
        correct = 0

        batches = [
            eval_examples[i: i + batch_size]
            for i in range(0, len(eval_examples), batch_size)
        ]

        for batch in batches:
            enc = collate_messages(
                batch, self.tokenizer, max_length=max_prompt_length,
                device=str(device),
            )
            out_ids = self.policy.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_response_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            responses = decode_responses(out_ids, enc["input_ids"], self.tokenizer)
            for resp, ex in zip(responses, batch):
                if is_correct(resp, ex["answer"]):
                    correct += 1

        return correct / len(eval_examples) if eval_examples else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_log_probs_from_logits(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token log-probs masked to response positions."""
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = response_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs * shift_mask
