from collections import deque
import threading

import numpy as np
import requests
import torch
from torch import Tensor

from lerobot.utils.messaging import pack_msg, unpack_msg
from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_remote import RemoteConfig


class RemotePolicy(PreTrainedPolicy):
    """
    A policy that proxies inference to a remote HTTP server.
    """

    config_class = RemoteConfig
    name = "remote"

    def __init__(self, config: RemoteConfig):
        super().__init__(config)
        self.server_url = config.server_url.rstrip("/")
        self.timeout = config.timeout
        self._thread_state = threading.local()
        self.reset()

    def get_optim_params(self) -> dict:
        return {}

    def reset(self):
        # Reinitialize thread-local state so each worker gets its own queue/session
        self._thread_state = threading.local()

    def _state(self):
        state = self._thread_state
        if not hasattr(state, "session"):
            state.session = requests.Session()
        if not hasattr(state, "action_queue"):
            state.action_queue = deque(maxlen=self.config.n_action_steps)
        return state

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict] | tuple[Tensor, None]:
        raise NotImplementedError("RemotePolicy is inference-only")

    def custom_prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch.pop('action')
        batch.pop('next.reward')
        batch.pop('next.done')
        batch.pop('next.truncated')
        batch.pop('info')
        task = batch.pop('task')

        batch['observation.task_instr'] = task

        if not hasattr(self, 'previous_state'):
            self.previous_state = batch["observation.state"].clone()

        batch["observation.state"] = torch.stack([self.previous_state, batch["observation.state"]], dim=1)
        self.previous_state = batch["observation.state"][:, -1].clone()

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        state = self._state()
        batch = self.custom_prepare_batch(batch)

        # Build payload with raw tensors/arrays; pack_msg handles encoding
        add_args = self.config.additional_args or {}
        payload = batch | add_args

        packed = pack_msg(payload)

        last_exception = None
        for _ in range(self.config.attempts):
            try:
                resp = state.session.post(
                    f"{self.server_url}/predict",
                    data=packed,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                last_exception = e

        if last_exception:
            raise last_exception

        unpacked = unpack_msg(resp.content)
        if isinstance(unpacked, torch.Tensor):
            actions = unpacked
        else:
            actions_np = np.asarray(unpacked)
            actions = torch.from_numpy(actions_np)

        device = torch.device(self.config.device)
        return actions.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        queue = self._state().action_queue

        if len(queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            queue.extend(actions.transpose(0, 1))  # [(B, A)] x T

        return queue.popleft()
