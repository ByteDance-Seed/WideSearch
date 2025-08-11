# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import functools
import time
from traceback import format_exc
from typing import Any, Iterable, List, Optional, Union

from loguru import logger
from openai import AzureOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from volcenginesdkarkruntime import Ark

from src.agent.schema import LLMOutputItem, ModelResponse, ToolCall
from src.utils.config import model_config


def retry(func, max_retries=8, init_interval=8):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        wait_time = init_interval
        while retries < max_retries:
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(f"Exception occurred: {e}:\n{format_exc()}")
                # If error message contains "maximum context length", do not retry
                if (
                    hasattr(e, "args")
                    and e.args
                    and "maximum context length" in str(e.args[0])
                ):
                    logger.warning("Maximum context length exceeded, not retrying.")
                    break
                if (
                    "maximum context length" in str(e)
                    or "maximum length" in str(e)
                    or "max message tokens" in str(e)
                ):
                    logger.warning("Maximum context length exceeded, not retrying.")
                    break
                retries += 1
                logger.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time += init_interval  # Increase wait time for each retry
        raise Exception(f"Function {func.__name__} failed after {max_retries} retries.")

    return wrapper


@retry
def ark_complete(
    base_url: str,
    api_key: Optional[str],
    messages: List[dict],
    model_name: str,
    tools: Optional[List[dict]] = None,
    **generate_kwargs,
) -> Optional[ChatCompletionMessage]:
    def create_ark_client(base_url, api_key):
        return Ark(
            base_url=base_url,
            api_key=api_key,
        )

    ark_client = create_ark_client(base_url, api_key)
    logger.info(f"messages: {messages}")
    logger.debug(f"tools: {tools}")
    logger.debug(generate_kwargs)
    completion = ark_client.chat.completions.create(
        messages=messages,  # type: ignore
        tools=tools,  # type: ignore
        model=model_name,
        **generate_kwargs,
    )
    logger.info(f"completion: {completion}")
    try:
        message = completion.choices[0].message  # type: ignore
        return message  # type: ignore
    except Exception as e:
        logger.warning(f"Error during completion: {e}")
        return None


@retry
def openai_complete(
    base_url: str,
    api_key: Optional[str],
    messages: Iterable[dict],
    tools: Optional[Iterable[dict]] = None,
    model_name: str = "gpt-4o-2024-05-13",
    retry_if_empty: bool = False,
    **generate_kwargs,
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with OpenAI APIs."""

    def create_openai_client(base_url, api_key):
        return AzureOpenAI(
            api_version="2023-03-15-preview",
            azure_endpoint=base_url,
            api_key=api_key,
            timeout=300,
        )

    openai_client = create_openai_client(base_url, api_key)
    logger.debug(f"messages: {messages}")
    logger.debug(f"tools: {tools}")
    logger.debug(generate_kwargs)

    completion = openai_client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_name,
        tools=tools,  # type: ignore
        **generate_kwargs,
    )
    message = None

    try:
        message = completion.choices[0].message
    except Exception as e:
        logger.warning(f"Error during completion: {e}")
        return None

    if retry_if_empty and not message.content and not message.tool_calls:
        raise RuntimeError(
            "[openai_complete] Got message, but content and toolcalls is empty, retry"
        )

    return message


@retry
def claude_complete(
    base_url: str,
    api_key: Optional[str],
    messages: Iterable[dict],
    tools: Optional[Iterable[dict]] = None,
    model_name: str = "aws_claude35_sonnet",
    **generate_kwargs,
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with claude APIs."""

    def create_claude_client(base_url, api_key):
        return AzureOpenAI(
            api_version="2023-09-06-preview",
            azure_endpoint=base_url,
            api_key=api_key,
            timeout=300,
            default_headers={"caller": "noname"},
        )

    claude_client = create_claude_client(base_url, api_key)
    logger.info(f"messages: {messages}")
    completion = claude_client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_name,
        tools=tools,  # type: ignore
        **generate_kwargs,
    )
    try:
        message = completion.choices[0].message
        tool_calls = message.tool_calls or []
        for choice in completion.choices[1:]:
            if choice.message.tool_calls:
                tool_calls.extend(choice.message.tool_calls)
        if tool_calls:
            message.tool_calls = tool_calls
        return message
    except Exception as e:
        logger.warning(f"Error during completion: {e}")
        return None


def get_is_claude_thinking(model_config_name: str) -> bool:
    assert (
        model_config_name in model_config
    ), f"model_config_name {model_config_name} not found in model_config"
    return model_config[model_config_name].get("is_claude_thinking", False)


def get_default_system_prompt_insert(model_config_name: str) -> str:
    return model_config[model_config_name].get("default_system_prompt", "")


def llm_completion(
    messages: Union[str, List[dict]],
    tools: Optional[List[dict]] = None,
    model_config_name: str = "doubao-1.6",
) -> Optional[ChatCompletionMessage]:
    """Complete a prompt with given LLM, raise error if the request failed."""

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    assert (
        model_config_name in model_config
    ), f"model_config_name {model_config_name} not found in model_config"
    model_name = model_config[model_config_name]["model_name"]
    base_url = model_config[model_config_name]["base_url"]
    api_key = model_config[model_config_name]["api_key"]
    generate_kwargs = model_config[model_config_name].get("generate_kwargs", {})

    logger.debug(
        f"model_config_name: {model_config_name}, model_name: {model_name}, generate_kwargs: {generate_kwargs}"
    )

    if "doubao" in model_name or model_name.startswith("ep") or "k2" in model_name:
        response = ark_complete(
            base_url=base_url,
            api_key=api_key,
            messages=messages,
            model_name=model_name,
            tools=tools,
            **generate_kwargs,
        )
    elif (
        "gpt" in model_name
        or "o3" in model_name
        or "gemini" in model_name
        or "o4" in model_name
    ):
        retry_if_empty = True if "gemini" in model_name else False
        response = openai_complete(
            base_url=base_url,
            api_key=api_key,
            messages=messages,
            tools=tools,
            model_name=model_name,
            retry_if_empty=retry_if_empty,
            **generate_kwargs,
        )
    elif "claude" in model_name:
        response = claude_complete(
            base_url=base_url,
            api_key=api_key,
            messages=messages,
            tools=tools,
            model_name=model_name,
            **generate_kwargs,
        )

    return response


def transform_model_response(response: Any | None) -> ModelResponse:
    out = ModelResponse()
    if response is None:
        out.error_marker = {"message": "Calling LLM failed."}
        return out

    # Set fields.
    item = LLMOutputItem(content=response.content)
    # Convert into dict to get optional fields.
    resp_dict = response.model_dump()
    if resp_dict.get("reasoning_content"):
        item.reasoning_content = resp_dict["reasoning_content"]
    if resp_dict.get("signature"):
        item.signature = resp_dict["signature"]

    if response.tool_calls:
        item.tool_calls = []
        for tool_call in response.tool_calls:
            item.tool_calls.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    # TODO: Randomly generate the ID if not provided.
                    tool_call_id=tool_call.id,
                )
            )
    out.outputs.append(item)
    return out
