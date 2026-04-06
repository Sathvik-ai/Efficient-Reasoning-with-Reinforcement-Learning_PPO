"""Model and tokenizer loading utilities."""

import logging
import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig

logger = logging.getLogger(__name__)


def load_tokenizer(cfg: ModelConfig) -> AutoTokenizer:
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        padding_side="left",   # left-pad for batched generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token ('%s')", tokenizer.eos_token)
    return tokenizer


def load_model(
    cfg: ModelConfig,
    device: Optional[str] = None,
) -> AutoModelForCausalLM:
    """Load the causal LM and optionally wrap it with LoRA."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.torch_dtype, torch.bfloat16)

    model_kwargs = dict(
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    # Use flash attention if requested and available
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    if device is not None and device != "auto":
        model_kwargs["device_map"] = device
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        **model_kwargs,
    )

    if cfg.use_lora:
        model = _apply_lora(model, cfg)

    return model


def _apply_lora(model: AutoModelForCausalLM, cfg: ModelConfig) -> AutoModelForCausalLM:
    """Wrap the model with LoRA adapters (requires the peft library)."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install it with: pip install peft"
        ) from exc

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_reference_model(
    cfg: ModelConfig,
    device: Optional[str] = None,
) -> AutoModelForCausalLM:
    """Load a frozen reference (SFT) model for KL-divergence computation."""
    ref_model = load_model(cfg, device=device)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()
    logger.info("Reference model loaded and frozen.")
    return ref_model


def get_model_and_tokenizer(
    cfg: ModelConfig,
    device: Optional[str] = None,
    load_reference: bool = True,
) -> Tuple[AutoModelForCausalLM, Optional[AutoModelForCausalLM], AutoTokenizer]:
    """Convenience loader that returns ``(policy, reference, tokenizer)``."""
    tokenizer = load_tokenizer(cfg)
    policy = load_model(cfg, device=device)
    reference = load_reference_model(cfg, device=device) if load_reference else None
    return policy, reference, tokenizer


def save_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    merge_lora: bool = True,
) -> None:
    """Save the model (and optionally merge LoRA weights) to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    # If the model is a PeftModel, optionally merge and unload adapters first
    try:
        from peft import PeftModel  # type: ignore

        if isinstance(model, PeftModel) and merge_lora:
            logger.info("Merging LoRA adapters before saving…")
            model = model.merge_and_unload()
    except ImportError:
        pass  # peft not installed – nothing to merge

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to '%s'.", output_dir)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
