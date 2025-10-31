from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("remote")
@dataclass
class RemoteConfig(PreTrainedConfig):
    # Identity and device placement
    type: str = field(default="remote", metadata={"help": "Policy type name"})
    device: str = field(default="cpu", metadata={"help": "Device used for returned tensors"})

    # Action execution
    # How many environment steps to execute per policy call. Used by the runtime action queue.
    n_action_steps: int = field(default=1, metadata={"help": "Number of env steps to execute per call"})

    # Remote-specific
    server_url: str = field(default="http://localhost:8000", metadata={"help": "Remote policy server URL"})
    timeout: float = field(default=30.0, metadata={"help": "HTTP timeout in seconds"})
    attempts: int = field(default=1, metadata={"help": "Number of retry attempts for failed requests"})

    # Additional arguments to inject directly into the observation dict (e.g. {"inference_config": {...}})
    additional_args: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Extra observation keys to inject directly into observation"},
    )

    # --- Abstract API implementations required by PreTrainedConfig ---
    def get_optimizer_preset(self) -> AdamWConfig:
        """Remote policy is inference-only; return a inert preset for API compatibility."""
        return AdamWConfig(lr=1e-5, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self):
        # No scheduler needed for inference-only policy
        return None

    def validate_features(self) -> None:
        # Minimal validation: allow any combination, but require at least one input feature
        if not self.input_features:
            raise ValueError("RemoteConfig requires at least one input feature to be defined.")

    @property
    def observation_delta_indices(self):
        # No temporal deltas required for observations by default
        return None

    @property
    def action_delta_indices(self):
        # Minimal behavior: align deltas to n_action_steps
        return list(range(self.n_action_steps))

    @property
    def reward_delta_indices(self):
        return None