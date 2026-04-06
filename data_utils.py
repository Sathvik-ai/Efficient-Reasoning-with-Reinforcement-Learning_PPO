"""Data loading and preprocessing utilities."""

from typing import Any, Dict, List, Optional, Tuple

from config import DataConfig


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

GSM8K_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think through each problem step-by-step, "
    "then write your final answer after '####'."
)


def build_prompt(
    question: str,
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
) -> str:
    """Return a chat-formatted prompt string."""
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{question}\n"
        "<|assistant|>\n"
    )


def build_chat_messages(
    question: str,
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Return a list of chat message dicts suitable for tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(solution: str) -> str:
    """Extract the numeric answer from a GSM8K solution string."""
    import re

    match = re.search(r"####\s*(.+)", solution)
    return match.group(1).strip() if match else solution.strip()


def load_gsm8k(
    cfg: DataConfig,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load and preprocess the GSM8K dataset.

    Returns:
        train_examples: list of dicts with keys ``prompt``, ``answer``.
        eval_examples: list of dicts with keys ``prompt``, ``answer``.
    """
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset(cfg.dataset_name)
    train_split = dataset[cfg.dataset_split]
    eval_split = dataset[cfg.eval_dataset_split]

    def _preprocess(example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "prompt": build_prompt(example["question"], cfg.system_prompt),
            "messages": build_chat_messages(example["question"], cfg.system_prompt),
            "answer": extract_gsm8k_answer(example["answer"]),
        }

    train_examples = [_preprocess(ex) for ex in train_split]
    eval_examples = [_preprocess(ex) for ex in eval_split]

    if cfg.num_train_samples is not None:
        train_examples = train_examples[: cfg.num_train_samples]
    if cfg.num_eval_samples is not None:
        eval_examples = eval_examples[: cfg.num_eval_samples]

    return train_examples, eval_examples


def load_math_dataset(
    cfg: DataConfig,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load and preprocess the MATH dataset (Hendrycks et al.)."""
    import re

    from datasets import load_dataset  # type: ignore

    dataset = load_dataset(cfg.dataset_name)
    train_split = dataset[cfg.dataset_split]
    eval_split = dataset[cfg.eval_dataset_split]

    def _extract_boxed(solution: str) -> str:
        match = re.search(r"\\boxed\{([^}]+)\}", solution)
        return match.group(1).strip() if match else solution.strip()

    def _preprocess(example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "prompt": build_prompt(example["problem"], cfg.system_prompt),
            "messages": build_chat_messages(example["problem"], cfg.system_prompt),
            "answer": _extract_boxed(example.get("solution", "")),
        }

    train_examples = [_preprocess(ex) for ex in train_split]
    eval_examples = [_preprocess(ex) for ex in eval_split]

    if cfg.num_train_samples is not None:
        train_examples = train_examples[: cfg.num_train_samples]
    if cfg.num_eval_samples is not None:
        eval_examples = eval_examples[: cfg.num_eval_samples]

    return train_examples, eval_examples


def get_dataset(
    cfg: DataConfig,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Dispatch to the correct dataset loader based on ``cfg.dataset_name``."""
    name = cfg.dataset_name.lower()
    if "gsm8k" in name:
        return load_gsm8k(cfg)
    if "math" in name or "hendrycks" in name:
        return load_math_dataset(cfg)
    raise ValueError(
        f"Unsupported dataset '{cfg.dataset_name}'. "
        "Supported datasets: gsm8k, math (hendrycks)."
    )


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def batch_examples(
    examples: List[Dict[str, Any]],
    batch_size: int,
) -> List[List[Dict[str, Any]]]:
    """Yield successive batches from a list of examples."""
    return [
        examples[i: i + batch_size]
        for i in range(0, len(examples), batch_size)
    ]


def collate_prompts(
    examples: List[Dict[str, str]],
    tokenizer,
    max_length: int,
    device: str = "cuda",
):
    """Tokenize a batch of prompt strings.

    Returns a dict of tensors ready for ``model.generate()``.
    """
    prompts = [ex["prompt"] for ex in examples]
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in encoding.items()}


def collate_messages(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    device: str = "cuda",
):
    """Apply chat template and tokenize a batch of message lists.

    Returns a dict of tensors ready for ``model.generate()``.
    """
    prompts = [
        tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in examples
    ]
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in encoding.items()}


def decode_responses(
    output_ids,
    input_ids,
    tokenizer,
) -> List[str]:
    """Decode only the newly generated tokens (excluding the prompt)."""
    generated = output_ids[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)
