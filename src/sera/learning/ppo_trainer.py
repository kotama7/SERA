"""PPO Trainer with LoRA-only updates per section 9.3.

Implements Proximal Policy Optimisation for fine-tuning the LoRA adapter
used by the agent LLM.  The trainer operates on batches of
:class:`PPORollout` instances collected during tree search, computes GAE
advantages, and performs clipped surrogate updates on only the LoRA
parameters.

Library usage:
- ``trl.trainer.utils.entropy_from_logits`` for entropy computation
- ``accelerate.Accelerator`` for device management and gradient ops
- ``peft.get_peft_model_state_dict`` for adapter weight extraction
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from sera.learning.rollout import PPORollout, PPORolloutV2, PPORolloutV3

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO trainer that only updates LoRA parameters.

    Uses established libraries for core operations:

    1. Collect rollouts as ``(prompt, response, reward)`` batches.
    2. Compute advantages using Generalised Advantage Estimation (GAE).
    3. PPO clipping loss with ``clip_range``.
    4. Value-function loss (MSE) with trl's value head.
    5. Entropy bonus via ``trl.trainer.utils.entropy_from_logits``.
    6. Gradient clipping via ``accelerate.Accelerator``.
    7. Adaptive KL coefficient control.
    8. Only update LoRA parameters (filter ``requires_grad``).
    """

    def __init__(
        self,
        exec_spec: Any,
        model_spec: Any,
        lineage_manager: Any,
        log_path: Path,
        plan_spec: Any = None,
    ) -> None:
        from sera.utils.logging import JsonlLogger

        self.exec_spec = exec_spec
        self.model_spec = model_spec
        self.lineage_manager = lineage_manager
        self.plan_spec = plan_spec
        self.logger = JsonlLogger(log_path)

        # Extract learning hyperparameters with sensible defaults
        learning = getattr(exec_spec, "learning", exec_spec)
        self.clip_range: float = getattr(learning, "clip_range", 0.2)
        self.lr: float = getattr(learning, "lr", 1e-4)
        self.batch_size: int = getattr(learning, "batch_size", 16)
        self.mini_batch_size: int = getattr(learning, "mini_batch_size", 4)
        self.epochs_per_update: int = getattr(learning, "epochs_per_update", 4)
        self.gamma: float = getattr(learning, "gamma", 0.99)
        self.gae_lambda: float = getattr(learning, "gae_lambda", 0.95)
        self.kl_control: bool = getattr(learning, "kl_control", True)
        self.kl_coef: float = getattr(learning, "kl_coef", 0.01)
        self.kl_target: float = getattr(learning, "kl_target", 0.01)
        self.entropy_coef: float = getattr(learning, "entropy_coef", 0.01)
        self.max_grad_norm: float = getattr(learning, "max_grad_norm", 0.5)
        self.value_loss_coef: float = getattr(learning, "value_loss_coef", 0.5)
        self.ppo_trigger_interval: int = getattr(learning, "ppo_trigger_interval", 5)

        # Mock function for testing without a real model
        self._mock_fn: Any = None

        # Plateau detection state
        self._best_lcb: float | None = None
        self._steps_since_improvement: int = 0

    # ------------------------------------------------------------------
    # Testing hook
    # ------------------------------------------------------------------

    def set_mock(self, mock_fn: Any) -> None:
        """Inject a mock: ``mock_fn(rollouts) -> dict`` with update stats."""
        self._mock_fn = mock_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(
        self,
        rollouts: list[PPORollout],
        agent_llm: Any,
        specs: Any,
        all_nodes: dict | None = None,
    ) -> dict:
        """Run one PPO update cycle on the collected rollouts.

        Parameters
        ----------
        rollouts : list[PPORollout]
            Batch of rollout episodes.
        agent_llm : AgentLLM
            Agent LLM instance (used for log-prob computation in full mode).
        specs : object
            Placeholder for additional spec references.
        all_nodes : dict | None
            Search tree nodes for resolving parent adapter IDs.

        Returns
        -------
        dict
            Update statistics including ``mean_reward``,
            ``kl_divergence``, ``policy_loss``, ``value_loss``,
            ``entropy``, and ``delta_norm_l2``.
        """
        if self._mock_fn is not None:
            result = self._mock_fn(rollouts)
            self.logger.log({"event": "ppo_update", **result})
            return result

        # Route advantage computation based on reward method
        self._compute_advantages_for_method(rollouts)

        return await self._ppo_update_impl(rollouts, agent_llm, specs, all_nodes=all_nodes)

    def notify_step(self, current_best_lcb: float) -> None:
        """Notify the trainer of the current best LCB for plateau detection."""
        if self._best_lcb is None or current_best_lcb > self._best_lcb:
            self._best_lcb = current_best_lcb
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

    def should_update(self, n_evaluated: int, evaluated_nodes: list | None = None) -> bool:
        """Return ``True`` when enough nodes have been evaluated to trigger
        a PPO update, or when a plateau is detected.

        Does **not** trigger if all evaluated nodes are constraint-violated
        (no valid reward signal).
        """
        if n_evaluated < 2:
            return False
        # Do not update if all evaluated nodes are constraint-violated
        if evaluated_nodes is not None and len(evaluated_nodes) > 0:
            all_violated = all(not getattr(n, "feasible", True) for n in evaluated_nodes)
            if all_violated:
                return False
        if n_evaluated % self.ppo_trigger_interval == 0:
            return True
        # Plateau trigger
        plateau_patience = getattr(getattr(self.exec_spec, "termination", None), "plateau_patience", 10)
        if self._steps_since_improvement >= plateau_patience:
            return True
        return False

    # ------------------------------------------------------------------
    # Method-based advantage routing
    # ------------------------------------------------------------------

    def _compute_advantages_for_method(self, rollouts: list[PPORollout]) -> None:
        """Route advantage computation based on the reward method in plan_spec."""
        method = "outcome_rm"
        if self.plan_spec is not None:
            reward_cfg = getattr(self.plan_spec, "reward", None)
            method = getattr(reward_cfg, "method", "outcome_rm") if reward_cfg else "outcome_rm"

        if method == "hiper":
            hiper_cfg = getattr(self.plan_spec, "hiper", None) if self.plan_spec else None
            if hiper_cfg is not None:
                from sera.learning.hierarchical_ppo import HierarchicalAdvantageEstimator

                estimator = HierarchicalAdvantageEstimator(hiper_cfg)
                # Build turn_rewards_map from PPORolloutV2 instances
                turn_rewards_map: dict[str, dict[str, float]] = {}
                for r in rollouts:
                    if isinstance(r, PPORolloutV2) and r.turn_rewards:
                        turn_rewards_map[r.node_id] = r.turn_rewards
                estimator.compute_hierarchical_advantages(rollouts, turn_rewards_map)
                return
            # Fallback to GAE if no hiper config
            logger.warning("HiPER method selected but no hiper config; falling back to GAE")

        elif method == "tool_aware":
            from sera.learning.tool_usage_learning import ToolCallRecord, compute_reward_tool_aware

            for r in rollouts:
                tool_records = []
                if isinstance(r, PPORolloutV3) and r.tool_trajectory:
                    for entry in r.tool_trajectory:
                        if isinstance(entry, ToolCallRecord):
                            tool_records.append(entry)
                        elif isinstance(entry, dict):
                            tool_records.append(ToolCallRecord(
                                tool_name=entry.get("tool_name", ""),
                                success=entry.get("success", False),
                                latency_sec=entry.get("wall_time_sec", 0.0),
                            ))
                r.reward = compute_reward_tool_aware(r.reward, tool_records)
            self._compute_gae(rollouts, self.gamma, self.gae_lambda)
            return

        # outcome_rm and mt_grpo both use standard GAE
        self._compute_gae(rollouts, self.gamma, self.gae_lambda)

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gae(
        rollouts: list[PPORollout],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalised Advantage Estimation in-place.

        Each rollout is treated as an independent single-step episode
        (no bootstrapping from a successor state), so:

            delta_t = reward - value
            advantage = delta_t
            returns = reward
        """
        for r in rollouts:
            delta = r.reward - r.value
            r.advantage = delta
            r.returns = r.reward

    # ------------------------------------------------------------------
    # Full PPO implementation (requires PyTorch + local model)
    # ------------------------------------------------------------------

    async def _ppo_update_impl(
        self,
        rollouts: list[PPORollout],
        agent_llm: Any,
        specs: Any,
        all_nodes: dict | None = None,
    ) -> dict:
        """Actual PPO implementation using trl/peft/accelerate primitives."""
        import numpy as np
        import torch
        from accelerate import Accelerator
        from peft import get_peft_model_state_dict
        from trl.trainer.utils import entropy_from_logits

        # Sleep vLLM engine to free GPU memory for PPO training
        vllm_engine = getattr(agent_llm, "_vllm_engine", None)
        if vllm_engine is not None:
            vllm_engine.sleep()

        try:
            # OOM retry with batch halving (up to 2 retries)
            effective_batch_size = self.batch_size
            max_oom_retries = 2
            last_exc = None
            for oom_attempt in range(max_oom_retries + 1):
                try:
                    original_batch_size = self.batch_size
                    self.batch_size = effective_batch_size
                    result = await self._ppo_update_core(
                        rollouts,
                        agent_llm,
                        specs,
                        np=np,
                        torch=torch,
                        Accelerator=Accelerator,
                        get_peft_model_state_dict=get_peft_model_state_dict,
                        entropy_from_logits=entropy_from_logits,
                        all_nodes=all_nodes,
                    )
                    self.batch_size = original_batch_size
                    return result
                except RuntimeError as e:
                    self.batch_size = original_batch_size
                    if "out of memory" in str(e).lower() and oom_attempt < max_oom_retries:
                        effective_batch_size = max(1, effective_batch_size // 2)
                        logger.warning(
                            "GPU OOM during PPO update, halving batch to %d (attempt %d/%d)",
                            effective_batch_size,
                            oom_attempt + 1,
                            max_oom_retries,
                        )
                        torch.cuda.empty_cache()
                        last_exc = e
                    else:
                        raise
            raise last_exc  # type: ignore[misc]
        finally:
            # Wake vLLM engine regardless of success/failure
            if vllm_engine is not None:
                vllm_engine.wake()

    async def _ppo_update_core(
        self,
        rollouts: list[PPORollout],
        agent_llm: Any,
        specs: Any,
        *,
        np: Any,
        torch: Any,
        Accelerator: Any,
        get_peft_model_state_dict: Any,
        entropy_from_logits: Any,
        all_nodes: dict | None = None,
    ) -> dict:
        """Core PPO logic, separated for sleep/wake wrapping."""

        # Step 1: fill value-head estimates and recompute advantages
        for r in rollouts:
            if r.value == 0.0 and hasattr(agent_llm, "get_value"):
                r.value = agent_llm.get_value(r.prompt, r.response)
        self._compute_advantages_for_method(rollouts)

        # Step 2: identify LoRA parameters
        model = agent_llm._model
        if model is None:
            raise RuntimeError("Local model not initialised for PPO update")

        # Validate LoRA compatibility before update
        try:
            from sera.specs.model_spec import validate_lora_compatibility

            model_config_dict = {
                "hidden_size": getattr(model.config, "hidden_size", 0),
                "num_attention_heads": getattr(model.config, "num_attention_heads", 0),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", 0),
                "model_type": getattr(model.config, "model_type", ""),
            }
            lora_config_dict = {
                "rank": getattr(self, "lora_rank", getattr(getattr(self.exec_spec, "learning", None), "rank", 16)),
                "alpha": getattr(self, "lora_alpha", getattr(getattr(self.exec_spec, "learning", None), "alpha", 32)),
                "target_modules": getattr(self, "target_modules", []),
            }
            is_compat, issues = validate_lora_compatibility(model_config_dict, lora_config_dict)
            if not is_compat:
                logger.warning("LoRA compatibility issues: %s", issues)
        except Exception as e:
            logger.debug("LoRA compatibility check skipped: %s", e)

        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]
        if not lora_params:
            logger.warning("No LoRA parameters found; skipping PPO update")
            return {
                "mean_reward": float(np.mean([r.reward for r in rollouts])),
                "kl_divergence": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "delta_norm_l2": 0.0,
            }

        # Use accelerate for device management and gradient ops
        accelerator = Accelerator()
        optimizer = torch.optim.AdamW(lora_params, lr=self.lr)
        model, optimizer = accelerator.prepare(model, optimizer)

        # Snapshot pre-update weights via peft API
        peft_model = getattr(model, "pretrained_model", model)
        pre_weights = {k: v.clone().detach() for k, v in get_peft_model_state_dict(peft_model).items()}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        bs = min(self.mini_batch_size, len(rollouts))

        for _epoch in range(self.epochs_per_update):
            indices = np.random.permutation(len(rollouts))
            for start in range(0, len(rollouts), bs):
                batch_idx = indices[start : start + bs]
                batch = [rollouts[int(i)] for i in batch_idx]

                # Forward pass: new log-probs with logits for entropy
                new_log_probs = []
                batch_logits = []
                for r in batch:
                    lp, logits = agent_llm.get_log_probs_with_logits(r.prompt, r.response)
                    new_log_probs.append(lp)
                    # Mean entropy across response tokens for this sample
                    batch_logits.append(logits)

                new_lp = torch.tensor(new_log_probs, dtype=torch.float32)
                old_lp = torch.tensor([r.log_prob for r in batch], dtype=torch.float32)
                advantages = torch.tensor([r.advantage for r in batch], dtype=torch.float32)
                returns_t = torch.tensor([r.returns for r in batch], dtype=torch.float32)
                values_t = torch.tensor([r.value for r in batch], dtype=torch.float32)

                # Normalise advantages
                if len(advantages) > 1:
                    adv_std = advantages.std()
                    if adv_std > 1e-8:
                        advantages = (advantages - advantages.mean()) / adv_std

                # PPO clipped surrogate
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = 0.5 * ((values_t - returns_t) ** 2).mean()

                # Entropy via trl's tested implementation (replaces buggy manual approx)
                entropy_sum = torch.tensor(0.0)
                for logit_tensor in batch_logits:
                    # entropy_from_logits: (batch, seq, vocab) -> (batch, seq)
                    ent = entropy_from_logits(logit_tensor)
                    entropy_sum = entropy_sum + ent.mean()
                entropy = entropy_sum / len(batch_logits)

                # Approximate KL
                approx_kl = (old_lp - new_lp).mean().item()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # NaN/Inf detection: skip this update if loss is invalid
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf detected in PPO loss, skipping update")
                    continue

                optimizer.zero_grad()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(lora_params, self.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl
                n_updates += 1

        # Post-update: compute delta norm via peft API
        post_weights = {k: v.clone().detach() for k, v in get_peft_model_state_dict(peft_model).items()}
        delta_norm = sum((post_weights[k] - pre_weights[k]).norm().item() ** 2 for k in pre_weights) ** 0.5

        # Save delta to lineage tree (section 9.3 step 6-7)
        new_adapter_node_id: str | None = None
        if self.lineage_manager is not None and delta_norm > 0:
            delta_tensors = {k: post_weights[k] - pre_weights[k] for k in pre_weights}
            new_adapter_node_id = str(uuid.uuid4())

            # Determine parent adapter_node_id from the best rollout's node (§6.7:
            # "PPO更新対象のノード群の「最良ノード」の親LoRAをベースに更新")
            parent_adapter_id = "adapter_root"
            if rollouts and all_nodes:
                best_rollout = max(rollouts, key=lambda r: r.reward)
                best_node = all_nodes.get(best_rollout.node_id)
                if best_node and getattr(best_node, "parent_id", None):
                    parent_node = all_nodes.get(best_node.parent_id)
                    if parent_node and getattr(parent_node, "adapter_node_id", None):
                        parent_adapter_id = parent_node.adapter_node_id

            adapter_spec_hash = getattr(
                getattr(self.model_spec, "compatibility", None),
                "adapter_spec_hash",
                "",
            )

            # Compute depth from parent's lineage path
            adapter_depth = 0
            if parent_adapter_id and parent_adapter_id != "adapter_root":
                parent_meta = self.lineage_manager.get_meta(parent_adapter_id)
                if parent_meta is not None:
                    adapter_depth = parent_meta.get("depth", 0) + 1
            elif parent_adapter_id == "adapter_root":
                adapter_depth = 1

            try:
                self.lineage_manager.save_delta(
                    adapter_node_id=new_adapter_node_id,
                    parent_id=parent_adapter_id,
                    delta_tensors=delta_tensors,
                    search_node_id=best_rollout.node_id if rollouts else "",
                    depth=adapter_depth,
                    adapter_spec_hash=adapter_spec_hash,
                )
                logger.info(
                    "Saved adapter delta %s (norm=%.4f)",
                    new_adapter_node_id[:8],
                    delta_norm,
                )
            except Exception as e:
                logger.error("Failed to save adapter delta: %s", e)
                new_adapter_node_id = None

        # Approximate aggregate KL
        mean_kl = total_kl / max(n_updates, 1)

        # Adaptive KL coefficient
        if self.kl_control:
            if mean_kl > self.kl_target * 1.5:
                self.kl_coef *= 2.0
            elif mean_kl < self.kl_target / 1.5:
                self.kl_coef /= 2.0

        mean_reward = float(np.mean([r.reward for r in rollouts]))

        # Collect turn_rewards from V2 rollouts
        all_turn_rewards: dict[str, list[float]] = {}
        for r in rollouts:
            if hasattr(r, "turn_rewards") and r.turn_rewards:
                for phase_key, value in r.turn_rewards.items():
                    all_turn_rewards.setdefault(phase_key, []).append(value)
        avg_turn_rewards = {k: float(np.mean(v)) for k, v in all_turn_rewards.items()} if all_turn_rewards else {}

        result = {
            "mean_reward": mean_reward,
            "kl_divergence": mean_kl,
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "delta_norm_l2": delta_norm,
            "kl_coef_current": self.kl_coef,
            "new_adapter_node_id": new_adapter_node_id,
            "turn_rewards": avg_turn_rewards,
        }

        self.logger.log({"event": "ppo_update", **result})
        return result
