"""Reward functions for reasoning tasks."""

import re
from typing import List, Optional

from config import RewardConfig


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the final answer from a model response.

    Looks for common answer delimiters:
      - ``#### <answer>``  (GSM8K style)
      - ``\\boxed{<answer>}``  (MATH style)
      - ``The answer is <answer>``
    """
    # GSM8K style: #### 42
    gsm8k_match = re.search(r"####\s*(.+)", text)
    if gsm8k_match:
        return gsm8k_match.group(1).strip()

    # LaTeX boxed: \boxed{42}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Natural language: "The answer is 42"
    natural_match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)", text
    )
    if natural_match:
        return natural_match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, removes commas in numbers, and lowercases.
    """
    answer = answer.strip().lower()
    # Remove commas from numbers (e.g. "1,000" -> "1000")
    answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
    # Remove trailing punctuation
    answer = answer.rstrip(".")
    return answer


def is_correct(prediction: str, ground_truth: str) -> bool:
    """Return True if the extracted prediction matches the ground truth."""
    pred_answer = extract_final_answer(prediction)
    if pred_answer is None:
        return False
    return normalize_answer(pred_answer) == normalize_answer(ground_truth)


def has_reasoning_steps(response: str) -> bool:
    """Return True if the response contains at least minimal chain-of-thought."""
    # Simple heuristic: check for step indicators or multi-sentence reasoning
    step_patterns = [
        r"[Ss]tep\s+\d+",
        r"[Ff]irst",
        r"[Ss]econd",
        r"[Tt]hird",
        r"[Ff]inally",
        r"[Tt]herefore",
        r"[Ss]o\s+",
        r"[Bb]ecause",
    ]
    return any(re.search(p, response) for p in step_patterns)


class RuleBasedRewardFunction:
    """Assigns rewards based on correctness and response quality heuristics."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg

    def __call__(
        self,
        responses: List[str],
        ground_truths: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        rewards: List[float] = []
        for response, truth in zip(responses, ground_truths):
            reward = 0.0

            # Correctness
            if is_correct(response, truth):
                reward += self.cfg.correct_answer_reward
            else:
                reward += self.cfg.wrong_answer_penalty

            # Format bonus: reward for visible reasoning steps
            if has_reasoning_steps(response):
                reward += self.cfg.format_reward

            # Length penalty (penalize overly long responses)
            if self.cfg.length_penalty_coef > 0:
                num_tokens = len(response.split())
                reward -= self.cfg.length_penalty_coef * num_tokens

            rewards.append(reward)
        return rewards


class RewardModelFunction:
    """Assigns rewards using a trained reward model."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load the reward model to avoid import overhead."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.reward_model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.reward_model_name,
            torch_dtype=torch.float16,
        )
        self._model.eval()

    def __call__(
        self,
        responses: List[str],
        ground_truths: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        if self._model is None:
            self._load()

        import torch

        prompts = prompts or [""] * len(responses)
        rewards: List[float] = []
        batch_size = self.cfg.reward_model_batch_size

        for i in range(0, len(responses), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            batch_responses = responses[i: i + batch_size]
            texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            scores = outputs.logits[:, 0].tolist()
            rewards.extend(scores)

        return rewards


def build_reward_function(cfg: RewardConfig):
    """Factory: return the appropriate reward function for the given config."""
    if cfg.reward_type == "rule_based":
        return RuleBasedRewardFunction(cfg)
    if cfg.reward_type == "reward_model":
        return RewardModelFunction(cfg)
    raise ValueError(
        f"Unknown reward_type '{cfg.reward_type}'. "
        "Choose 'rule_based' or 'reward_model'."
    )


def normalize_rewards(rewards: List[float], clip: float = 5.0) -> List[float]:
    """Normalize rewards to zero mean, unit variance, then clip."""
    if len(rewards) < 2:
        return rewards

    import statistics

    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) or 1.0
    normalized = [(r - mean) / std for r in rewards]
    return [max(-clip, min(clip, r)) for r in normalized]
