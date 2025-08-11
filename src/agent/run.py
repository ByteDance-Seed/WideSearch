# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
from typing import Literal

from loguru import logger

from src.agent.agent import DEFAULT_MAX_ERROR_COUNT, DEFAULT_MAX_STEPS, Agent
from src.agent.memory import (
    ActionStep,
    ActionStepError,
    MemoryAgent,
    StepStatus,
    UserInputStep,
)
from src.agent.prompt import get_system_prompt
from src.agent.schema import (
    ErrorMarker,
    ModelResponse,
    RunResult,
    ToolCall,
    ToolCallResult,
)
from src.agent.tools import _default_tools
from src.utils.llm import (
    get_default_system_prompt_insert,
    get_is_claude_thinking,
    llm_completion,
    transform_model_response,
)


class Runner:
    @classmethod
    async def run(
        cls,
        starting_agent: Agent,
        user_input: str,
        memory: MemoryAgent | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        llm_error_strategy: Literal["retry", "stop"] = "retry",
    ):
        # These variables are used to track the LLM error.
        llm_error_counter = 0
        # break_on_llm_error = False
        last_llm_error_message = ""

        # Initialize memory if not provided.
        if memory is None:
            memory = MemoryAgent(system_instructions=starting_agent.instructions)
        # If passed memory has no system instructions, use the starting agent's instructions.
        if memory.system_instructions is None:
            memory.system_instructions = starting_agent.instructions

        # Insert current user input into memory, which will add a new turn for the memory.
        last_turn = memory.insert_user_input(user_input)

        max_steps_summary = False
        # Iterate through steps until the agent stops or reaches the maximum number of turns.
        while not max_steps_summary and not last_turn.is_finished():
            # Stop if llm error counter reaches the maximum number.
            if (
                llm_error_strategy == "stop"
                and llm_error_counter > 0
                or llm_error_counter >= DEFAULT_MAX_ERROR_COUNT
            ):
                last_turn.steps.append(
                    UserInputStep(
                        user_input=f"[{last_llm_error_message}] Too many errors have occurred. Please stop invoking the tool immediately and answer the user's question."
                    )
                )
                max_steps_summary = True

            if last_turn.step_number >= max_steps:
                last_turn.steps.append(
                    UserInputStep(
                        user_input="[Max Step] The tool has been used too many times. Please stop invoking the tool immediately and answer the user's question."
                    )
                )
                max_steps_summary = True
            # Execute one step
            output_action_step = await cls._step(
                agent=starting_agent,
                memory=memory,
            )

            # If the output_action_step is an ActionStepError, increment the llm_error_counter.
            # And continue to the next step(Not to yield anything).
            if isinstance(output_action_step, ActionStepError):
                llm_error_counter += 1
                last_llm_error_message = output_action_step.message
                continue

            # Yield the result of this step
            yield output_action_step

    @classmethod
    async def run_until_stop(
        cls,
        starting_agent: Agent,
        user_input: str,
        memory: MemoryAgent | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        llm_error_strategy: Literal["retry", "stop"] = "retry",
    ):
        llm_error_counter = 0

        # TODO: merge `run` and `run_until_stop` into one function.
        # Initialize memory if not provided.
        if memory is None:
            memory = MemoryAgent(system_instructions=starting_agent.instructions)
        # If passed memory has no system instructions, use the starting agent's instructions.
        if memory.system_instructions is None:
            memory.system_instructions = starting_agent.instructions

        # Insert current user input into memory, which will add a new turn for the memory.
        last_turn = memory.insert_user_input(user_input)

        # Iterate through steps until the agent stops or reaches the maximum number of turns.
        while last_turn.step_number < max_steps and not last_turn.is_finished():
            # Stop if llm error counter reaches the maximum number.
            if llm_error_strategy == "stop" and llm_error_counter > 0:
                logging.warning("[Runner] stop because `llm_error_strategy=stop`")
                break
            elif llm_error_counter >= DEFAULT_MAX_ERROR_COUNT:
                logging.warning(
                    f"[Runner] LLM error counter reaches the maximum number: {DEFAULT_MAX_ERROR_COUNT}"
                )
                break

            # Execute one step
            _output_action_step = await cls._step(
                agent=starting_agent,
                memory=memory,
            )

            # If the output_action_step is an ActionStepError, increment the llm_error_counter.
            # And continue to the next step(Not to yield anything).
            if isinstance(_output_action_step, ActionStepError):
                llm_error_counter += 1
                continue

        if not last_turn.steps:
            return RunResult(stop_reason="error")

        # TODO: handle the case when llm error at last.

        last_step = last_turn.steps[-1]
        if isinstance(last_step, ActionStep):
            if last_step.error_marker is not None:
                message = last_step.error_marker["message"]
                return RunResult(stop_reason="error", content=message)
            if last_step.step_status == StepStatus.FINISHED:
                return RunResult(
                    stop_reason="finished",
                    content=last_step.content,
                    reasoning_content=last_step.reasoning_content,
                )
            elif last_step.step_status == StepStatus.CONTINUE:
                return RunResult(
                    stop_reason="reach_max_steps",
                    content=last_step.content,
                    reasoning_content=last_step.reasoning_content,
                )
            else:
                raise RuntimeError("[UnreachableCode]")
        else:
            return RunResult(stop_reason="error")

    @classmethod
    async def _invoke_tool_call(
        cls, agent: Agent, model_response: ModelResponse
    ) -> list[ToolCallResult]:
        assert model_response.outputs
        # For now only consider there is only one sample(a.k.a the first output).
        resp = model_response.outputs[0]

        async def _call(tool_call: ToolCall):
            tool_name = tool_call.tool_name
            tool = agent.get_tool_by_name(tool_name)
            if tool is None:
                logging.warning(
                    f"[Runner] Tool {tool_name} not found, skip this tool call."
                )
                return ToolCallResult(
                    tool_call_id=tool_call.tool_call_id,
                    error_marker=ErrorMarker(message=f"Tool {tool_name} not found"),
                )

            tool_call_id = tool_call.tool_call_id
            arguments = tool_call.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logging.warning(
                        f"[Runner] Failed to parse tool call arguments: {arguments}"
                    )
                    arguments = {}

            try:
                response = await tool(**arguments)
            except Exception as e:
                logging.warning(f"[Runner] Failed to invoke tool {tool_name}: {e}")
                return ToolCallResult(
                    tool_call_id=tool_call_id,
                    error_marker=ErrorMarker(message=str(e)),
                )
            error_marker = (
                ErrorMarker(message=response.error) if response.error else None
            )
            system_error_marker = (
                ErrorMarker(message=response.system_error)
                if response.system_error
                else None
            )
            tool_call_result = ToolCallResult(
                tool_call_id=tool_call_id,
                content=response.data,
                error_marker=error_marker,
                system_error_marker=system_error_marker,
                extra=response.extra,
            )
            return tool_call_result

        tasks = [_call(tc) for tc in resp.tool_calls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    @classmethod
    def _to_action_step(
        cls,
        agent: Agent,
        model_response: ModelResponse,
        tool_call_results: list[ToolCallResult],
    ) -> ActionStep:
        assert model_response.outputs
        # For now only consider there is only one sample(a.k.a the first output).
        resp = model_response.outputs[0]

        step_status = StepStatus.CONTINUE if tool_call_results else StepStatus.FINISHED

        action_step = ActionStep(
            step_status=step_status,
            content=resp.content,
            reasoning_content=resp.reasoning_content,
            signature=resp.signature,
            tool_calls=resp.tool_calls,
            tool_call_results=tool_call_results,
        )
        return action_step

    @classmethod
    async def _step(
        cls, *, agent: Agent, memory: MemoryAgent
    ) -> ActionStep | ActionStepError:
        try:
            model_response = transform_model_response(
                llm_completion(
                    messages=memory.to_message(
                        is_claude_thinking=get_is_claude_thinking(
                            agent.model_config_name
                        ),
                        default_system_prompt_insert=get_default_system_prompt_insert(
                            agent.model_config_name
                        ),
                    ),
                    tools=agent.tools_desc,
                    model_config_name=agent.model_config_name,
                )
            )
        except Exception as e:
            # If there is an exception during LLM completion, return an ActionStepError.
            # And this step will not be tracked in the memory.
            message = f"[Runner] Exception during LLM completion: {e}"
            return ActionStepError(message=message)

        if model_response.error_marker is not None or not model_response.outputs:
            # Case when LLM returns an error or no output.
            # Just set an action step with error marker.
            error_marker = model_response.error_marker or {
                "message": "No output from model."
            }
            action_step = ActionStep(error_marker=error_marker)
        else:
            tool_call_results = await cls._invoke_tool_call(
                agent=agent, model_response=model_response
            )

            action_step = cls._to_action_step(
                agent=agent,
                model_response=model_response,
                tool_call_results=tool_call_results,
            )

        # To avoid the memory being modified by the caller, we use deepcopy here.
        memory.insert_action_step(deepcopy(action_step))

        return action_step


async def run_turn(agent, memory, user_input):
    r = Runner.run(agent, memory=memory, user_input=user_input)
    async for step in r:
        logger.debug(f"step: {step}")


def extract_messages_from_memory(memory: MemoryAgent, skip_tools=False):
    def custom_asdict_factory(data):
        def convert_value(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return dict((k, convert_value(v)) for k, v in data)

    messages = []
    for turn in memory.turns:
        for step in turn.steps:
            if isinstance(step, UserInputStep):
                messages.append({"role": "user", "content": step.user_input})
            else:
                content = asdict(step, dict_factory=custom_asdict_factory)
                messages.append({"role": "assistant", "content": content})
    return messages


async def run_single_query(
    query: str,
    agent_name: str = "",
    model_config_name: str = "",
    tools: dict = _default_tools,
    tools_desc: list[dict] = [],
    system_prompt: str = get_system_prompt(language="zh"),
):
    agent = Agent(
        name=agent_name,
        tools=tools,
        tools_desc=tools_desc,
        model_config_name=model_config_name,
    )
    logger.debug(f"query: {query}")
    logger.debug(f"system_prompt: {system_prompt}")

    memory = MemoryAgent(system_instructions=system_prompt)
    logger.info(f"agent running: {agent_name}, model: {model_config_name}.")
    await run_turn(agent, memory, query)
    logger.info(f"agent finished: {agent_name}, model: {model_config_name}.")
    return extract_messages_from_memory(memory)
