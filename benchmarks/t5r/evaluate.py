from __future__ import annotations

from typing import Callable

from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason  # type: ignore[import-not-found]
from tau2.data_model.tasks import RewardType, Task  # type: ignore[import-not-found]
from tau2.environment.environment import Environment  # type: ignore[import-not-found]
from tau2.environment.toolkit import get_tool_types  # type: ignore[import-not-found]
from tau2.evaluator.evaluator_action import ActionEvaluator  # type: ignore[import-not-found]
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator  # type: ignore[import-not-found]
from tau2.evaluator.evaluator_env import EnvironmentEvaluator  # type: ignore[import-not-found]


def evaluate_text_simulation(
    *,
    simulation: SimulationRun,
    task: Task,
    environment_constructor: Callable[..., Environment],
) -> RewardInfo:
    """Evaluate a half-duplex tau2 simulation without importing tau2.registry."""
    termination = simulation.termination_reason
    if isinstance(termination, str):
        termination_value = termination
    else:
        termination_value = termination.value
    if termination_value not in {
        TerminationReason.AGENT_STOP.value,
        TerminationReason.USER_STOP.value,
    }:
        return RewardInfo(
            reward=0.0,
            reward_basis=None,
            info={"note": f"Simulation terminated prematurely: {termination_value}"},
        )
    if task.evaluation_criteria is None:
        return RewardInfo(reward=1.0, reward_basis=None, info={"note": "No criteria"})

    trajectory = simulation.messages or []
    env = environment_constructor()
    tool_types = get_tool_types(env.tools) if env.tools is not None else None
    env_reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=environment_constructor,
        task=task,
        full_trajectory=trajectory,
        solo_mode=False,
    )
    action_reward = ActionEvaluator.calculate_reward(
        task=task,
        full_trajectory=trajectory,
        tool_types=tool_types,
    )
    communicate_reward = CommunicateEvaluator.calculate_reward(
        task=task,
        full_trajectory=trajectory,
    )

    reward = 1.0
    reward_breakdown = {}
    basis = set(task.evaluation_criteria.reward_basis)
    if basis & {RewardType.DB, RewardType.ENV_ASSERTION}:
        if env_reward.reward_breakdown:
            reward_breakdown.update(env_reward.reward_breakdown)
        reward *= env_reward.reward
    if RewardType.ACTION in basis:
        if action_reward.reward_breakdown:
            reward_breakdown.update(action_reward.reward_breakdown)
        reward *= action_reward.reward
    if RewardType.COMMUNICATE in basis:
        if communicate_reward.reward_breakdown:
            reward_breakdown.update(communicate_reward.reward_breakdown)
        reward *= communicate_reward.reward
    if RewardType.NL_ASSERTION in basis:
        return RewardInfo(
            reward=0.0,
            reward_basis=task.evaluation_criteria.reward_basis,
            info={"note": "NL assertions are not part of the T5-R tau3 text pilot evaluator"},
        )

    return RewardInfo(
        reward=reward,
        db_check=env_reward.db_check,
        env_assertions=env_reward.env_assertions,
        action_checks=action_reward.action_checks,
        communicate_checks=communicate_reward.communicate_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=reward_breakdown,
        info={
            "env": env_reward.info,
            "communicate": communicate_reward.info,
            "action": action_reward.info,
        },
    )
