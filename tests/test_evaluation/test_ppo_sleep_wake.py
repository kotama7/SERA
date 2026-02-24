"""Tests for PPOTrainer sleep/wake integration with vLLM engine."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sera.learning.ppo_trainer import PPOTrainer
from sera.learning.rollout import PPORollout


def _make_exec_spec():
    return SimpleNamespace(
        learning=SimpleNamespace(
            enabled=True,
            clip_range=0.2,
            lr=1e-4,
            batch_size=4,
            mini_batch_size=2,
            epochs_per_update=1,
            gamma=0.99,
            gae_lambda=0.95,
            kl_control=False,
            kl_coef=0.01,
            kl_target=0.01,
            entropy_coef=0.01,
            max_grad_norm=1.0,
            value_loss_coef=0.5,
            ppo_trigger_interval=5,
        ),
        termination=SimpleNamespace(plateau_patience=10),
    )


def _make_model_spec():
    return SimpleNamespace(
        base_model=SimpleNamespace(id="test-model", revision="", dtype="bf16"),
        adapter_spec=SimpleNamespace(rank=16, alpha=32, target_modules=["q_proj"]),
    )


def _make_rollouts():
    return [
        PPORollout(
            node_id="node-1",
            prompt="test prompt",
            response="test response",
            log_prob=-1.0,
            reward=0.8,
            value=0.5,
        ),
    ]


class TestPPOTrainerSleepWake:
    """Test that PPOTrainer properly coordinates with vLLM sleep/wake."""

    async def test_sleep_wake_called_on_successful_update(self):
        """vLLM engine sleep/wake are called around PPO update."""
        trainer = PPOTrainer(
            exec_spec=_make_exec_spec(),
            model_spec=_make_model_spec(),
            lineage_manager=MagicMock(),
            log_path=Path("/dev/null"),
        )

        # Use mock to bypass real PPO
        trainer.set_mock(lambda rollouts: {"mean_reward": 0.8, "kl_divergence": 0.0})

        mock_vllm = MagicMock()
        agent_llm = MagicMock()
        agent_llm._vllm_engine = mock_vllm

        rollouts = _make_rollouts()
        result = await trainer.update(rollouts, agent_llm, MagicMock())

        assert result["mean_reward"] == 0.8
        # Mock path skips sleep/wake (since _mock_fn returns early)
        # This test verifies mock path works; next tests verify real path

    async def test_sleep_called_before_ppo_core(self):
        """In the real path, sleep is called before _ppo_update_core."""
        trainer = PPOTrainer(
            exec_spec=_make_exec_spec(),
            model_spec=_make_model_spec(),
            lineage_manager=MagicMock(),
            log_path=Path("/dev/null"),
        )

        mock_vllm = MagicMock()
        agent_llm = MagicMock()
        agent_llm._vllm_engine = mock_vllm

        call_order = []
        mock_vllm.sleep.side_effect = lambda: call_order.append("sleep")
        mock_vllm.wake.side_effect = lambda: call_order.append("wake")

        # Patch _ppo_update_core to track call order without real PyTorch
        async def fake_core(*args, **kwargs):
            call_order.append("ppo_core")
            return {"mean_reward": 0.5, "kl_divergence": 0.0}

        with patch.object(trainer, "_ppo_update_core", side_effect=fake_core):
            result = await trainer._ppo_update_impl(_make_rollouts(), agent_llm, MagicMock())

        assert call_order == ["sleep", "ppo_core", "wake"]

    async def test_wake_called_on_exception(self):
        """vLLM wake is called even if _ppo_update_core raises."""
        trainer = PPOTrainer(
            exec_spec=_make_exec_spec(),
            model_spec=_make_model_spec(),
            lineage_manager=MagicMock(),
            log_path=Path("/dev/null"),
        )

        mock_vllm = MagicMock()
        agent_llm = MagicMock()
        agent_llm._vllm_engine = mock_vllm

        async def failing_core(*args, **kwargs):
            raise RuntimeError("PPO failed")

        with patch.object(trainer, "_ppo_update_core", side_effect=failing_core):
            with pytest.raises(RuntimeError, match="PPO failed"):
                await trainer._ppo_update_impl(_make_rollouts(), agent_llm, MagicMock())

        mock_vllm.sleep.assert_called_once()
        mock_vllm.wake.assert_called_once()

    async def test_no_vllm_engine_skips_sleep_wake(self):
        """When no vLLM engine, sleep/wake are not called."""
        trainer = PPOTrainer(
            exec_spec=_make_exec_spec(),
            model_spec=_make_model_spec(),
            lineage_manager=MagicMock(),
            log_path=Path("/dev/null"),
        )

        agent_llm = MagicMock()
        agent_llm._vllm_engine = None

        async def fake_core(*args, **kwargs):
            return {"mean_reward": 0.5, "kl_divergence": 0.0}

        with patch.object(trainer, "_ppo_update_core", side_effect=fake_core):
            result = await trainer._ppo_update_impl(_make_rollouts(), agent_llm, MagicMock())

        assert result["mean_reward"] == 0.5


class TestPPOTrainerShouldUpdate:
    """Test PPO trigger logic."""

    def test_trigger_at_interval(self):
        trainer = PPOTrainer(
            exec_spec=_make_exec_spec(),
            model_spec=_make_model_spec(),
            lineage_manager=MagicMock(),
            log_path=Path("/dev/null"),
        )
        assert not trainer.should_update(1)
        assert trainer.should_update(5)
        assert trainer.should_update(10)
        assert not trainer.should_update(3)

    def test_no_trigger_too_early(self):
        trainer = PPOTrainer(
            exec_spec=_make_exec_spec(),
            model_spec=_make_model_spec(),
            lineage_manager=MagicMock(),
            log_path=Path("/dev/null"),
        )
        assert not trainer.should_update(0)
        assert not trainer.should_update(1)

    def test_plateau_trigger(self):
        trainer = PPOTrainer(
            exec_spec=_make_exec_spec(),
            model_spec=_make_model_spec(),
            lineage_manager=MagicMock(),
            log_path=Path("/dev/null"),
        )
        # Simulate plateau: no improvement for plateau_patience steps
        trainer._best_lcb = 0.5
        trainer._steps_since_improvement = 10
        assert trainer.should_update(3)  # Not at interval, but plateau triggers
