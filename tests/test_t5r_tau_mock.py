from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAU_SOURCE = ROOT / ".external" / "tau2-bench"


@unittest.skipUnless(TAU_SOURCE.exists(), "pinned Tau source is not installed")
class TauMockIntegrationTests(unittest.TestCase):
    def test_official_runner_executes_mock_without_provider_calls(self) -> None:
        from benchmarks.t5r.tau_runtime import activate_tau_source, run_official_text_simulation

        activate_tau_source(TAU_SOURCE)
        try:
            from tau2.agent.base_agent import HalfDuplexAgent
            from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
            from tau2.domains.mock.environment import get_environment, get_tasks
            from tau2.orchestrator.orchestrator import Orchestrator
            from tau2.user.user_simulator_base import HalfDuplexUser, UserState
        except ModuleNotFoundError as exc:
            self.skipTest(f"Tau base dependencies are not installed: {exc}")

        class ScriptedUser(HalfDuplexUser[UserState]):
            def get_init_state(self, message_history=None):
                return UserState(system_messages=[], messages=list(message_history or []))

            def generate_next_message(self, message, state):
                state.messages.append(message)
                response = UserMessage(
                    role="user",
                    content="Create a task called Important Meeting for user_1.",
                )
                state.messages.append(response)
                return response, state

            def set_seed(self, seed):
                return None

        class ScriptedAgent(HalfDuplexAgent[int]):
            def get_init_state(self, message_history=None):
                return 0

            def generate_next_message(self, message, state):
                if isinstance(message, UserMessage):
                    return (
                        AssistantMessage(
                            role="assistant",
                            tool_calls=[
                                ToolCall(
                                    id="fixture-call",
                                    name="create_task",
                                    arguments={
                                        "user_id": "user_1",
                                        "title": "Important Meeting",
                                    },
                                )
                            ],
                        ),
                        1,
                    )
                if not isinstance(message, ToolMessage):
                    raise AssertionError(f"Expected ToolMessage, got {type(message)!r}")
                return AssistantMessage(role="assistant", content="Created. ###STOP###"), 2

            @classmethod
            def is_stop(cls, message):
                return bool(message.content and "###STOP###" in message.content)

            def set_seed(self, seed):
                return None

        environment = get_environment()
        task = next(task for task in get_tasks("base") if task.id == "create_task_1_with_env_assertions")
        agent = ScriptedAgent(environment.get_tools(), environment.get_policy())
        user = ScriptedUser(instructions=str(task.user_scenario), tools=None)
        simulation = run_official_text_simulation(
            Orchestrator(
                domain="mock",
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=6,
                seed=300,
                validate_communication=True,
            )
        )
        self.assertEqual(simulation.reward_info.reward, 1.0)
        self.assertEqual(simulation.termination_reason, "agent_stop")


if __name__ == "__main__":
    unittest.main()
