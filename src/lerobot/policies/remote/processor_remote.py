from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.policies.remote.configuration_remote import RemoteConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    RenameObservationsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_remote_pre_post_processors(
    config: RemoteConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    rename_map: dict[str, str] = {},
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Custom pre/post processors for the Remote policy.

    Pre:
      - Normalizer (if stats provided)
      - AddBatchDimension
      - AppendInferenceConfig (copies config.inference_config into the batch)
      - Device placement

    Post:
      - Device to CPU
      - Unnormalize outputs (if stats provided)
    """

    # Pre: allow renaming features and add batch dim. Rename map can be overridden at runtime
    # through preprocessor_overrides with the key "rename_observations_processor".
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map=rename_map),
        AddBatchDimensionProcessorStep(),
    ]

    # Minimal postprocessor: identity (no steps)
    output_steps: list[ProcessorStep] = []

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )