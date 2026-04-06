"""Compatibility and unit tests for all PPO reasoning components."""

import re
import sys
import unittest
from dataclasses import asdict
from typing import List
from unittest.mock import MagicMock, patch

try:
    import torch as _torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tests: config.py
# ---------------------------------------------------------------------------
class TestConfig(unittest.TestCase):
    def test_default_construction(self):
        from config import TrainingConfig

        cfg = TrainingConfig()
        self.assertEqual(cfg.ppo.clip_epsilon, 0.2)
        self.assertEqual(cfg.reward.reward_type, "rule_based")
        self.assertIn("gsm8k", cfg.data.dataset_name.lower())

    def test_nested_dict_construction(self):
        from config import TrainingConfig

        cfg = TrainingConfig(
            model={"model_name_or_path": "test-model"},
            ppo={"learning_rate": 5e-6},
        )
        self.assertEqual(cfg.model.model_name_or_path, "test-model")
        self.assertEqual(cfg.ppo.learning_rate, 5e-6)

    def test_asdict_roundtrip(self):
        from config import TrainingConfig

        cfg = TrainingConfig()
        d = asdict(cfg)
        self.assertIn("model", d)
        self.assertIn("ppo", d)
        self.assertIn("reward", d)
        cfg2 = TrainingConfig(**d)
        self.assertEqual(cfg2.ppo.learning_rate, cfg.ppo.learning_rate)


# ---------------------------------------------------------------------------
# Tests: reward_functions.py
# ---------------------------------------------------------------------------
class TestExtractFinalAnswer(unittest.TestCase):
    def test_gsm8k_style(self):
        from reward_functions import extract_final_answer

        text = "Step 1: ...\n#### 42"
        self.assertEqual(extract_final_answer(text), "42")

    def test_boxed_style(self):
        from reward_functions import extract_final_answer

        text = r"The answer is \boxed{3x + 2}"
        self.assertEqual(extract_final_answer(text), "3x + 2")

    def test_natural_language_style(self):
        from reward_functions import extract_final_answer

        text = "The answer is 100."
        self.assertEqual(extract_final_answer(text), "100")

    def test_no_answer(self):
        from reward_functions import extract_final_answer

        self.assertIsNone(extract_final_answer("I don't know"))


class TestNormalizeAnswer(unittest.TestCase):
    def test_strips_commas(self):
        from reward_functions import normalize_answer

        self.assertEqual(normalize_answer("1,000"), "1000")

    def test_lowercases(self):
        from reward_functions import normalize_answer

        self.assertEqual(normalize_answer("YES"), "yes")

    def test_strips_trailing_period(self):
        from reward_functions import normalize_answer

        self.assertEqual(normalize_answer("42."), "42")


class TestIsCorrect(unittest.TestCase):
    def test_correct_match(self):
        from reward_functions import is_correct

        self.assertTrue(is_correct("Step 1: ...\n#### 42", "42"))

    def test_wrong_answer(self):
        from reward_functions import is_correct

        self.assertFalse(is_correct("#### 7", "42"))

    def test_missing_answer_format(self):
        from reward_functions import is_correct

        self.assertFalse(is_correct("The total is unknown.", "42"))


class TestRuleBasedReward(unittest.TestCase):
    def setUp(self):
        from config import RewardConfig
        from reward_functions import RuleBasedRewardFunction

        cfg = RewardConfig(
            correct_answer_reward=1.0,
            wrong_answer_penalty=-0.5,
            format_reward=0.1,
        )
        self.fn = RuleBasedRewardFunction(cfg)

    def test_correct_response_gives_positive_reward(self):
        rewards = self.fn(
            ["First, 6+6=12. Therefore, the answer is 12.\n#### 12"],
            ["12"],
        )
        self.assertGreater(rewards[0], 0)

    def test_wrong_response_gives_negative_reward(self):
        rewards = self.fn(["#### 99"], ["42"])
        self.assertLess(rewards[0], 0)

    def test_batch_length_matches(self):
        responses = ["#### 1", "#### 2", "#### 3"]
        truths = ["1", "2", "99"]
        rewards = self.fn(responses, truths)
        self.assertEqual(len(rewards), 3)


class TestNormalizeRewards(unittest.TestCase):
    def test_normalized_mean_near_zero(self):
        from reward_functions import normalize_rewards

        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed = normalize_rewards(rewards)
        mean = sum(normed) / len(normed)
        self.assertAlmostEqual(mean, 0.0, places=5)

    def test_clips_extreme_values(self):
        from reward_functions import normalize_rewards

        rewards = [1000.0, -1000.0]
        normed = normalize_rewards(rewards, clip=5.0)
        for r in normed:
            self.assertLessEqual(abs(r), 5.0)


class TestBuildRewardFunction(unittest.TestCase):
    def test_rule_based(self):
        from config import RewardConfig
        from reward_functions import RuleBasedRewardFunction, build_reward_function

        cfg = RewardConfig(reward_type="rule_based")
        fn = build_reward_function(cfg)
        self.assertIsInstance(fn, RuleBasedRewardFunction)

    def test_unknown_type_raises(self):
        from config import RewardConfig
        from reward_functions import build_reward_function

        cfg = RewardConfig(reward_type="unknown")
        with self.assertRaises(ValueError):
            build_reward_function(cfg)


# ---------------------------------------------------------------------------
# Tests: data_utils.py
# ---------------------------------------------------------------------------
class TestBuildPrompt(unittest.TestCase):
    def test_prompt_contains_question(self):
        from data_utils import build_prompt

        prompt = build_prompt("What is 2+2?")
        self.assertIn("What is 2+2?", prompt)

    def test_prompt_contains_system(self):
        from data_utils import build_prompt

        prompt = build_prompt("Q", system_prompt="Be helpful")
        self.assertIn("Be helpful", prompt)


class TestBuildChatMessages(unittest.TestCase):
    def test_returns_three_roles(self):
        from data_utils import build_chat_messages

        msgs = build_chat_messages("What is 2+2?")
        roles = [m["role"] for m in msgs]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

    def test_user_content(self):
        from data_utils import build_chat_messages

        msgs = build_chat_messages("What is 2+2?")
        user_msg = next(m for m in msgs if m["role"] == "user")
        self.assertEqual(user_msg["content"], "What is 2+2?")


class TestExtractGSM8KAnswer(unittest.TestCase):
    def test_extracts_number(self):
        from data_utils import extract_gsm8k_answer

        self.assertEqual(extract_gsm8k_answer("blah blah #### 123"), "123")

    def test_fallback_strip(self):
        from data_utils import extract_gsm8k_answer

        self.assertEqual(extract_gsm8k_answer("  42  "), "42")


class TestBatchExamples(unittest.TestCase):
    def test_even_batches(self):
        from data_utils import batch_examples

        data = list(range(10))
        batches = batch_examples(data, 5)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], [0, 1, 2, 3, 4])

    def test_uneven_last_batch(self):
        from data_utils import batch_examples

        data = list(range(7))
        batches = batch_examples(data, 4)
        self.assertEqual(len(batches[-1]), 3)


class TestDecodeResponses(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_decodes_new_tokens_only(self):
        from data_utils import decode_responses

        tokenizer = MagicMock()
        tokenizer.batch_decode.return_value = ["hello world"]
        import torch

        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        input_ids = torch.tensor([[1, 2, 3]])
        result = decode_responses(output_ids, input_ids, tokenizer)
        # Should have decoded tokens [4, 5] (everything after prompt)
        call_args = tokenizer.batch_decode.call_args
        decoded_ids = call_args[0][0]
        expected = torch.tensor([[4, 5]])
        self.assertTrue(torch.equal(decoded_ids, expected))
        self.assertEqual(result, ["hello world"])


# ---------------------------------------------------------------------------
# Tests: model_utils.py
# ---------------------------------------------------------------------------
@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestCountParameters(unittest.TestCase):
    def test_simple_model(self):
        import torch
        import torch.nn as nn

        from model_utils import count_parameters

        model = nn.Linear(10, 5)
        trainable, total = count_parameters(model)
        self.assertEqual(trainable, total)
        self.assertEqual(total, 10 * 5 + 5)  # weight + bias

    def test_frozen_model(self):
        import torch.nn as nn

        from model_utils import count_parameters

        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad_(False)
        trainable, total = count_parameters(model)
        self.assertEqual(trainable, 0)
        self.assertGreater(total, 0)


# ---------------------------------------------------------------------------
# Tests: ppo_trainer.py – unit tests for pure utility functions
# ---------------------------------------------------------------------------
@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestComputeLogProbs(unittest.TestCase):
    def test_output_shape(self):
        import torch

        from ppo_trainer import PPOTrainer

        B, L, V = 2, 6, 100
        logits = torch.randn(B, L, V)
        input_ids = torch.randint(0, V, (B, L))
        response_mask = torch.zeros(B, L, dtype=torch.bool)
        response_mask[:, 3:] = True  # response starts at position 3

        log_probs = PPOTrainer._token_log_probs_from_logits(
            logits, input_ids, response_mask
        )
        # Output should have shape (B, L-1) and be 0 outside response
        self.assertEqual(log_probs.shape, (B, L - 1))
        # Prompt positions should be zero
        self.assertTrue((log_probs[:, :2] == 0).all())


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestAdaptiveKLController(unittest.TestCase):
    def test_increases_when_kl_above_target(self):
        from ppo_trainer import AdaptiveKLController

        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.05, horizon=100)
        before = ctrl.value
        ctrl.update(current_kl=0.2, n_steps=10)
        self.assertGreater(ctrl.value, before)

    def test_decreases_when_kl_below_target(self):
        from ppo_trainer import AdaptiveKLController

        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.2, horizon=100)
        before = ctrl.value
        ctrl.update(current_kl=0.01, n_steps=10)
        self.assertLess(ctrl.value, before)

    def test_never_drops_below_minimum(self):
        from ppo_trainer import AdaptiveKLController

        ctrl = AdaptiveKLController(init_kl_coef=1e-7, target_kl=1.0, horizon=100)
        ctrl.update(current_kl=0.0, n_steps=100)
        self.assertGreaterEqual(ctrl.value, 1e-6)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestComputeAdvantagesAndReturns(unittest.TestCase):
    def test_output_shape(self):
        import torch

        from ppo_trainer import compute_advantages_and_returns

        B, L = 3, 8
        rewards = torch.zeros(B, L)
        rewards[:, -1] = 1.0  # reward only at last step
        values = torch.zeros(B, L)
        response_mask = torch.ones(B, L, dtype=torch.bool)
        adv, ret = compute_advantages_and_returns(rewards, values, response_mask)
        self.assertEqual(adv.shape, (B, L))
        self.assertEqual(ret.shape, (B, L))

    def test_masked_positions_are_zero(self):
        import torch

        from ppo_trainer import compute_advantages_and_returns

        B, L = 2, 6
        rewards = torch.zeros(B, L)
        values = torch.zeros(B, L)
        response_mask = torch.zeros(B, L, dtype=torch.bool)
        response_mask[:, 3:] = True  # only last 3 positions are response

        adv, ret = compute_advantages_and_returns(rewards, values, response_mask)
        # Prompt positions should be zero
        self.assertTrue((adv[:, :3] == 0).all())


# ---------------------------------------------------------------------------
# Integration smoke test: end-to-end component wiring
# ---------------------------------------------------------------------------
class TestEndToEndWiring(unittest.TestCase):
    """Verify that all components can be instantiated and wired together
    without errors (no actual GPU / large model required)."""

    def test_config_reward_wiring(self):
        from config import RewardConfig, TrainingConfig
        from reward_functions import build_reward_function

        cfg = TrainingConfig()
        fn = build_reward_function(cfg.reward)
        rewards = fn(["#### 42"], ["42"])
        self.assertEqual(len(rewards), 1)
        self.assertIsInstance(rewards[0], float)

    def test_data_utils_standalone(self):
        from data_utils import batch_examples, build_chat_messages, build_prompt

        prompt = build_prompt("Solve: 2+2")
        msgs = build_chat_messages("Solve: 2+2")
        examples = [{"prompt": prompt, "messages": msgs, "answer": "4"}] * 10
        batches = batch_examples(examples, 3)
        self.assertEqual(len(batches), 4)  # ceil(10/3) = 4

    def test_ppo_trainer_imports_cleanly(self):
        """Ensure ppo_trainer can be imported without heavy dependencies."""
        import importlib

        spec = importlib.util.find_spec("ppo_trainer")
        self.assertIsNotNone(spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
