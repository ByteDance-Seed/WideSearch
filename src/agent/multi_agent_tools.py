# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List

from loguru import logger

from src.agent.run import run_single_query
from src.agent.tools import InternalResponse


@dataclass
class SubAgentInfo:
    index: int
    prompt: str
    response: str | None = None
    messages: List[dict] | None = None

    def to_dict(self):
        return {"index": self.index, "prompt": self.prompt, "response": self.response}


def create_sub_agents_wrap(
    agent_name, model_config_name, tools, tools_desc, system_prompt
) -> Callable[[List[dict]], Coroutine[Any, List[dict], InternalResponse]]:
    async def create_sub_agents(sub_agents: List[dict]) -> InternalResponse:
        logger.info(f"create sub agents, num: {len(sub_agents)}")
        logger.info(f"sub agents: {sub_agents}")
        try:
            new_sub_agents = [SubAgentInfo(**sub_agent) for sub_agent in sub_agents]
        except Exception as e:
            logger.error(f"create sub agents error: {e}")
            return InternalResponse(data=[{"error": f"create sub agents error: {e}"}])

        from concurrent.futures import ThreadPoolExecutor

        logger.debug(f"tools: {tools}")
        logger.debug(f"system_prompt: {system_prompt}")
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(
                lambda sub_agent: asyncio.run(
                    run_single_query(
                        sub_agent.prompt,
                        agent_name=agent_name,
                        model_config_name=model_config_name,
                        tools=tools,
                        tools_desc=tools_desc,
                        system_prompt=system_prompt,
                    )
                ),
                new_sub_agents,
            )
            _sub_agent = None
            try:
                for _sub_agent, result in zip(new_sub_agents, results):
                    logger.debug(
                        f"sub_agent: {_sub_agent}, result: {json.dumps(result, ensure_ascii=False)}"
                    )
                    _sub_agent.response = result[-1]["content"]["content"]
                    _sub_agent.messages = result
            except Exception:
                logger.error(f"run sub agents error: {traceback.format_exc()}")
                if _sub_agent:
                    _sub_agent.response = "sub_agent running error."
                    _sub_agent.messages = [
                        {
                            "content": {
                                "content": f"sub_agent running error: {traceback.format_exc()}"
                            }
                        }
                    ]

        return InternalResponse(
            data=json.dumps(
                [sub_agent.to_dict() for sub_agent in new_sub_agents],
                ensure_ascii=False,
            ),
            extra={"sub_agents": [sub_agent.messages for sub_agent in new_sub_agents]},
        )

    return create_sub_agents


def get_multi_agent_tools(
    model_name, model_config_name, sub_agent_tools, tool_desc, system_prompt
):
    tools = {
        "create_sub_agents": create_sub_agents_wrap(
            model_name, model_config_name, sub_agent_tools, tool_desc, system_prompt
        ),
    }
    tools.update(sub_agent_tools)
    return tools
