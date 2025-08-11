# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from types import CoroutineType
from typing import Any, Callable

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_ERROR_COUNT = 3


@dataclass
class Agent:
    """An agent is an AI model configured with instructions, tools, and more."""

    name: str
    """The name of the agent."""

    instructions: str | None = None
    """The instructions for the agent. Will be used as the "system prompt" when this agent is
    invoked. Describes what the agent should do, and how it responds."""

    tools: dict[str, Callable[..., CoroutineType]] = field(default_factory=dict)
    """A list of tools that the agent can use."""

    tools_desc: list[dict] = field(default_factory=list)
    """A list of tools description that the agent can use."""

    output_type: type[Any] | None = None
    """The type of the output object. If not provided, the output will be `str`."""

    model_config_name: str = "doubao-1.6"

    def get_tool_by_name(self, tool_name: str) -> Callable[..., CoroutineType] | None:
        """Return the tool by name."""
        return self.tools.get(tool_name)
