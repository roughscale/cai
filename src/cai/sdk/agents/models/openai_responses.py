from __future__ import annotations

import json
import os
import copy
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, AsyncStream, NotGiven
from openai.types import ChatModel
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ToolParam,
    WebSearchToolParam,
    response_create_params,
)

from .. import _debug
from ..agent_output import AgentOutputSchema
from ..exceptions import UserError
from ..handoffs import Handoff
from ..items import ItemHelpers, ModelResponse, TResponseInputItem
from ..logger import logger
from ..model_trace import write_model_trace
from ..run_to_jsonl import get_session_recorder
from ..tool import ComputerTool, FileSearchTool, FunctionTool, Tool, WebSearchTool
from ..tracing import SpanError, response_span
from ..usage import Usage
from ..version import __version__
from ..parallel_isolation import PARALLEL_ISOLATION
from ..simple_agent_manager import AGENT_MANAGER
from .openai_chatcompletions import set_current_active_model
from .interface import Model, ModelTracing

if TYPE_CHECKING:
    from ..model_settings import ModelSettings


_USER_AGENT = f"Agents/Python {__version__}"
_HEADERS = {"User-Agent": _USER_AGENT}

# From the Responses API
IncludeLiteral = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
]


class OpenAIResponsesModel(Model):
    """
    Implementation of `Model` that uses the OpenAI Responses API.
    """

    def __init__(
        self,
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
        agent_name: str = "Agent",
        agent_id: str | None = None,
        agent_type: str | None = None,
    ) -> None:
        self.model = model
        self._client = openai_client

        # Track interaction counter and token totals for cli display
        self.interaction_counter = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cost = 0.0
        self.agent_name = agent_name
        self.agent_type = agent_type or agent_name.lower().replace(" ", "_")
        self.uses_unified_context = False
        self.agent_id = agent_id or AGENT_MANAGER.get_agent_id()
        self._display_name = self.agent_name
        self.disable_rich_streaming = False
        self.suppress_final_output = False
        self.previous_response_id: str | None = None
        self.logger = get_session_recorder()

        if agent_id and PARALLEL_ISOLATION.is_parallel_mode():
            isolated_history = PARALLEL_ISOLATION.get_isolated_history(agent_id)
            self.message_history = isolated_history if isolated_history is not None else []
        else:
            existing_history = AGENT_MANAGER.get_message_history(self.agent_name)
            if existing_history is not None and isinstance(existing_history, list):
                self.message_history = existing_history
            else:
                self.message_history = []
                if self.agent_name not in AGENT_MANAGER._message_history:
                    AGENT_MANAGER._message_history[self.agent_name] = self.message_history

        if agent_id is not None and not PARALLEL_ISOLATION.is_parallel_mode():
            if self.agent_name in AGENT_MANAGER._message_history:
                self.message_history = AGENT_MANAGER._message_history[self.agent_name]

    def _log_user_input(self, input: str | list[TResponseInputItem]) -> None:
        """Log user-facing input items for session recording."""
        if not self.logger:
            return

        if isinstance(input, str):
            self.logger.log_user_message(input)
            return

        for item in input:
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue

            content = item.get("content")
            if isinstance(content, str):
                self.logger.log_user_message(content)
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"input_text", "output_text"}:
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                if parts:
                    self.logger.log_user_message("\n".join(parts))

    def _build_cli_message(self, response: Response):
        """Convert Responses output into the chat-like shape expected by the CLI renderer."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for content_item in item.content:
                    if isinstance(content_item, ResponseOutputText) and content_item.text:
                        text_parts.append(content_item.text)
                    elif isinstance(content_item, ResponseOutputRefusal) and content_item.refusal:
                        text_parts.append(content_item.refusal)
            elif isinstance(item, ResponseFunctionToolCall):
                tool_calls.append(
                    {
                        "id": item.call_id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": item.arguments or "{}",
                        },
                    }
                )

        return {
            "content": "\n".join(part for part in text_parts if part).strip(),
            "tool_calls": tool_calls,
        }

    def _log_assistant_output(self, response: Response) -> None:
        """Record assistant text/tool calls for Responses-backed runs."""
        if not self.logger:
            return

        cli_message = self._build_cli_message(response)
        content = cli_message["content"] or None
        tool_calls = cli_message["tool_calls"] or None
        if content is not None or tool_calls is not None:
            self.logger.log_assistant_message(content, tool_calls)

    def set_agent_name(self, name: str) -> None:
        """Set the agent name for CLI display purposes."""
        self.agent_name = name

    def add_to_message_history(self, msg):
        """Keep a chat-style history for CLI features and agent-manager interoperability."""
        is_duplicate = False

        if self.message_history:
            if msg.get("role") in ["system", "user"]:
                is_duplicate = any(
                    existing.get("role") == msg.get("role")
                    and existing.get("content") == msg.get("content")
                    for existing in self.message_history
                )
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_id = msg["tool_calls"][0].get("id")
                indices_to_remove = []
                for i, existing in enumerate(self.message_history):
                    if (
                        existing.get("role") == "assistant"
                        and existing.get("tool_calls")
                        and existing["tool_calls"][0].get("id") == tool_call_id
                    ):
                        indices_to_remove.append(i)
                for i in reversed(indices_to_remove):
                    self.message_history.pop(i)
                is_duplicate = False
            elif msg.get("role") == "tool":
                is_duplicate = any(
                    existing.get("role") == "tool"
                    and existing.get("tool_call_id") == msg.get("tool_call_id")
                    for existing in self.message_history
                )

        if not is_duplicate:
            self.message_history.append(msg)
            manager_history = AGENT_MANAGER.get_message_history(self.agent_name)
            if manager_history is not self.message_history:
                AGENT_MANAGER.add_to_history(self.agent_name, msg)
            if PARALLEL_ISOLATION.is_parallel_mode() and self.agent_id:
                PARALLEL_ISOLATION.update_isolated_history(self.agent_id, msg)

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value if value is not None else NOT_GIVEN

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        # Increment the interaction counter for CLI display
        self.interaction_counter += 1
        set_current_active_model(self)

        write_model_trace(
            "model_get_response_start",
            adapter="openai_responses",
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            model=str(self.model),
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tool_names=[tool.name for tool in tools if hasattr(tool, "name")],
            handoff_names=[handoff.tool_name for handoff in handoffs],
            output_schema=output_schema.json_schema() if output_schema else None,
        )
        
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                self._log_user_input(input)
                response = await self._fetch_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    stream=False,
                )

                if _debug.DONT_LOG_MODEL_DATA:
                    logger.debug("LLM responded")
                else:
                    logger.debug(
                        "LLM resp:\n"
                        f"{json.dumps([x.model_dump() for x in response.output], indent=2)}\n"
                    )

                usage = (
                    Usage(
                        requests=1,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                    if response.usage
                    else Usage()
                )

                if tracing.include_data():
                    span_response.span_data.response = response
                    span_response.span_data.input = input
                    
                # Print the agent message for CLI display
                from cai.util import cli_print_agent_messages, calculate_model_cost
                try:
                    interaction_cost = calculate_model_cost(
                        str(self.model), usage.input_tokens, usage.output_tokens
                    )
                    self.total_cost = (self.total_cost or 0.0) + (interaction_cost or 0.0)

                    message_obj = self._build_cli_message(response)
                    cli_print_agent_messages(
                        agent_name=getattr(self, 'agent_name', 'Agent'),
                        message=message_obj,
                        counter=getattr(self, 'interaction_counter', 0),
                        model=str(self.model),
                        debug=False,
                        interaction_input_tokens=usage.input_tokens,
                        interaction_output_tokens=usage.output_tokens,
                        interaction_reasoning_tokens=0,  # Not available in Responses API
                        total_input_tokens=getattr(self, 'total_input_tokens', 0),
                        total_output_tokens=getattr(self, 'total_output_tokens', 0),
                        total_reasoning_tokens=getattr(self, 'total_reasoning_tokens', 0),
                        interaction_cost=interaction_cost,
                        total_cost=self.total_cost,
                        suppress_empty=True,
                    )

                    # Update token totals
                    self.total_input_tokens += usage.input_tokens
                    self.total_output_tokens += usage.output_tokens
                except Exception as e:
                    logger.error(f"Error printing agent message: {e}")

                self._log_assistant_output(response)

                self.previous_response_id = response.id

                write_model_trace(
                    "model_get_response_end",
                    adapter="openai_responses",
                    agent_name=self.agent_name,
                    model=str(self.model),
                    usage=usage,
                    referenceable_id=response.id,
                    raw_response=response,
                    model_response_output=response.output,
                )

            except Exception as e:
                write_model_trace(
                    "model_get_response_error",
                    adapter="openai_responses",
                    agent_name=self.agent_name,
                    model=str(self.model),
                    error=str(e),
                )
                span_response.set_error(
                    SpanError(
                        message="Error getting response",
                        data={
                            "error": str(e) if tracing.include_data() else e.__class__.__name__,
                        },
                    )
                )
                request_id = e.request_id if isinstance(e, APIStatusError) else None
                logger.error(f"Error getting response: {e}. (request_id: {request_id})")
                raise

        return ModelResponse(
            output=response.output,
            usage=usage,
            referenceable_id=response.id,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """
        Yields a partial message as it is generated, as well as the usage information.
        """
        # Increment the interaction counter for CLI display
        self.interaction_counter += 1
        set_current_active_model(self)
        
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                self._log_user_input(input)
                stream = await self._fetch_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    stream=True,
                )

                final_response: Response | None = None

                async for chunk in stream:
                    if isinstance(chunk, ResponseCompletedEvent):
                        final_response = chunk.response
                    yield chunk

                if final_response and tracing.include_data():
                    span_response.span_data.response = final_response
                    span_response.span_data.input = input

                # Print the agent message for CLI display
                from cai.util import cli_print_agent_messages, calculate_model_cost
                try:
                    interaction_cost = calculate_model_cost(
                        str(self.model),
                        final_response.usage.input_tokens,
                        final_response.usage.output_tokens,
                    )
                    self.total_cost = (self.total_cost or 0.0) + (interaction_cost or 0.0)

                    message_obj = self._build_cli_message(final_response)
                    cli_print_agent_messages(
                        agent_name=getattr(self, 'agent_name', 'Agent'),
                        message=message_obj,
                        counter=getattr(self, 'interaction_counter', 0),
                        model=str(self.model),
                        debug=False,
                        interaction_input_tokens=final_response.usage.input_tokens,
                        interaction_output_tokens=final_response.usage.output_tokens,
                        interaction_reasoning_tokens=0,  # Not available in Responses API
                        total_input_tokens=getattr(self, 'total_input_tokens', 0),
                        total_output_tokens=getattr(self, 'total_output_tokens', 0),
                        total_reasoning_tokens=getattr(self, 'total_reasoning_tokens', 0),
                        interaction_cost=interaction_cost,
                        total_cost=self.total_cost,
                        suppress_empty=True,
                    )

                    # Update token totals
                    self.total_input_tokens += final_response.usage.input_tokens
                    self.total_output_tokens += final_response.usage.output_tokens
                except Exception as e:
                    logger.error(f"Error printing agent message: {e}")

                self._log_assistant_output(final_response)

                if final_response is not None:
                    self.previous_response_id = final_response.id

            except Exception as e:
                span_response.set_error(
                    SpanError(
                        message="Error streaming response",
                        data={
                            "error": str(e) if tracing.include_data() else e.__class__.__name__,
                        },
                    )
                )
                logger.error(f"Error streaming response: {e}")
                raise

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[True],
    ) -> AsyncStream[ResponseStreamEvent]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[False],
    ) -> Response: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[True] | Literal[False] = False,
    ) -> Response | AsyncStream[ResponseStreamEvent]:
        list_input = self._sanitize_responses_input(
            ItemHelpers.input_to_new_input_list(input)
        )
        previous_response_id = self.previous_response_id

        if previous_response_id:
            incremental_items = [
                item
                for item in list_input
                if isinstance(item, dict) and item.get("type") == "function_call_output"
            ]
            if incremental_items:
                list_input = incremental_items
            else:
                previous_response_id = None
                self.previous_response_id = None

        import dataclasses
        env_reasoning_effort = os.environ.get("CAI_REASONING_EFFORT")
        env_verbosity = os.environ.get("CAI_VERBOSITY")
        if env_reasoning_effort is not None and model_settings.reasoning_effort is None:
            model_settings = dataclasses.replace(
                model_settings, reasoning_effort=env_reasoning_effort
            )
        if env_verbosity is not None and model_settings.verbosity is None:
            model_settings = dataclasses.replace(
                model_settings, verbosity=env_verbosity
            )

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else NOT_GIVEN
        )

        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        converted_tools = Converter.convert_tools(tools, handoffs)
        response_format = Converter.get_response_format(output_schema)
        reasoning = Converter.get_reasoning(model_settings)
        text_config = Converter.get_text_config(response_format, model_settings)
        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            logger.debug(
                f"Calling LLM {self.model} with input:\n"
                f"{json.dumps(list_input, indent=2)}\n"
                f"Tools:\n{json.dumps(converted_tools.tools, indent=2)}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        write_model_trace(
            "model_api_request",
            adapter="openai_responses",
            agent_name=self.agent_name,
            model=str(self.model),
            request={
                "instructions": system_instructions,
                "input": list_input,
                "include": converted_tools.includes,
                "tools": converted_tools.tools,
                "temperature": model_settings.temperature,
                "top_p": model_settings.top_p,
                "truncation": model_settings.truncation,
                "max_output_tokens": model_settings.max_tokens,
                "tool_choice": tool_choice if tool_choice is not NOT_GIVEN else None,
                "parallel_tool_calls": None if parallel_tool_calls is NOT_GIVEN else parallel_tool_calls,
                "text": text_config if text_config is not NOT_GIVEN else None,
                "reasoning": reasoning if reasoning is not NOT_GIVEN else None,
                "store": model_settings.store,
                "previous_response_id": previous_response_id,
                "stream": stream,
            },
        )

        response = await self._client.responses.create(
            instructions=self._non_null_or_not_given(system_instructions),
            model=self.model,
            input=list_input,
            include=converted_tools.includes,
            tools=converted_tools.tools,
            temperature=self._non_null_or_not_given(model_settings.temperature),
            top_p=self._non_null_or_not_given(model_settings.top_p),
            truncation=self._non_null_or_not_given(model_settings.truncation),
            max_output_tokens=self._non_null_or_not_given(model_settings.max_tokens),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            reasoning=reasoning,
            stream=stream,
            extra_headers=_HEADERS,
            text=text_config,
            store=self._non_null_or_not_given(model_settings.store),
            previous_response_id=self._non_null_or_not_given(previous_response_id),
        )
        return response

    def _sanitize_responses_input(
        self,
        input_items: list[TResponseInputItem],
    ) -> list[TResponseInputItem]:
        sanitized: list[TResponseInputItem] = []

        for item in copy.deepcopy(input_items):
            if not isinstance(item, dict):
                sanitized.append(item)
                continue

            role = item.get("role")
            item_type = item.get("type")

            # Responses API reasoning items are output-only in this full-history replay mode.
            # They should not be resent unless we are chaining with previous_response_id.
            if item_type == "reasoning":
                continue

            if role in {"user", "system", "developer", "assistant"}:
                content = item.get("content")
                content_type = "output_text" if role == "assistant" else "input_text"

                if isinstance(content, str):
                    content_parts = [{"type": content_type, "text": content}]
                elif isinstance(content, list):
                    content_parts = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue

                        normalized_part = dict(part)
                        part_type = normalized_part.get("type")

                        if part_type == "text":
                            normalized_part["type"] = content_type
                            normalized_part.setdefault("text", normalized_part.pop("text", ""))
                        elif part_type == "input_text" and role == "assistant":
                            normalized_part["type"] = "output_text"
                        elif part_type == "output_text" and role != "assistant":
                            normalized_part["type"] = "input_text"

                        content_parts.append(normalized_part)
                else:
                    content_parts = []

                if role == "assistant":
                    if content_parts:
                        sanitized.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": content_parts,
                            }
                        )

                    for tool_call in item.get("tool_calls", []) or []:
                        function = tool_call.get("function", {})
                        sanitized.append(
                            {
                                "type": "function_call",
                                "call_id": tool_call.get("id", ""),
                                "name": function.get("name", "unknown_function"),
                                "arguments": function.get("arguments", "{}"),
                            }
                        )
                    continue

                sanitized.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content_parts,
                    }
                )
                continue

            if role == "tool":
                sanitized.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.get("tool_call_id", ""),
                        "output": item.get("content") or "",
                    }
                )
                continue

            if item_type == "message" and isinstance(item.get("content"), str):
                item["content"] = [{"type": "input_text", "text": item["content"]}]

            sanitized.append(item)

        return sanitized

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            # Determine API key
            api_key = os.getenv("ALIAS_API_KEY", os.getenv("OPENAI_API_KEY", "sk-alias-1234567890"))
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client


@dataclass
class ConvertedTools:
    tools: list[ToolParam]
    includes: list[IncludeLiteral]


class Converter:
    @classmethod
    def convert_tool_choice(
        cls, tool_choice: Literal["auto", "required", "none"] | str | None
    ) -> response_create_params.ToolChoice | NotGiven:
        if tool_choice is None:
            return NOT_GIVEN
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "none":
            return "none"
        elif tool_choice == "file_search":
            return {
                "type": "file_search",
            }
        elif tool_choice == "web_search_preview":
            return {
                "type": "web_search_preview",
            }
        elif tool_choice == "computer_use_preview":
            return {
                "type": "computer_use_preview",
            }
        else:
            return {
                "type": "function",
                "name": tool_choice,
            }

    @classmethod
    def get_response_format(
        cls, output_schema: AgentOutputSchema | None
    ) -> ResponseTextConfigParam | NotGiven:
        if output_schema is None or output_schema.is_plain_text():
            return NOT_GIVEN
        else:
            return {
                "format": {
                    "type": "json_schema",
                    "name": "final_output",
                    "schema": output_schema.json_schema(),
                    "strict": output_schema.strict_json_schema,
                }
            }

    @classmethod
    def get_reasoning(
        cls,
        model_settings: ModelSettings,
    ) -> response_create_params.Reasoning | NotGiven:
        if model_settings.reasoning_effort is None:
            return NOT_GIVEN

        return {"effort": model_settings.reasoning_effort}

    @classmethod
    def get_text_config(
        cls,
        response_format: ResponseTextConfigParam | NotGiven,
        model_settings: ModelSettings,
    ) -> ResponseTextConfigParam | NotGiven:
        if response_format is NOT_GIVEN and model_settings.verbosity is None:
            return NOT_GIVEN

        text_config: dict[str, Any] = {}
        if response_format is not NOT_GIVEN:
            text_config.update(response_format)
        if model_settings.verbosity is not None:
            text_config["verbosity"] = model_settings.verbosity
        return text_config

    @classmethod
    def convert_tools(
        cls,
        tools: list[Tool],
        handoffs: list[Handoff[Any]],
    ) -> ConvertedTools:
        converted_tools: list[ToolParam] = []
        includes: list[IncludeLiteral] = []

        computer_tools = [tool for tool in tools if isinstance(tool, ComputerTool)]
        if len(computer_tools) > 1:
            raise UserError(f"You can only provide one computer tool. Got {len(computer_tools)}")

        for tool in tools:
            converted_tool, include = cls._convert_tool(tool)
            converted_tools.append(converted_tool)
            if include:
                includes.append(include)

        for handoff in handoffs:
            converted_tools.append(cls._convert_handoff_tool(handoff))

        return ConvertedTools(tools=converted_tools, includes=includes)

    @classmethod
    def _convert_tool(cls, tool: Tool) -> tuple[ToolParam, IncludeLiteral | None]:
        """Returns converted tool and includes"""

        if isinstance(tool, FunctionTool):
            converted_tool: ToolParam = {
                "name": tool.name,
                "parameters": tool.params_json_schema,
                "strict": tool.strict_json_schema,
                "type": "function",
                "description": tool.description,
            }
            includes: IncludeLiteral | None = None
        elif isinstance(tool, WebSearchTool):
            ws: WebSearchToolParam = {
                "type": "web_search_preview",
                "user_location": tool.user_location,
                "search_context_size": tool.search_context_size,
            }
            converted_tool = ws
            includes = None
        elif isinstance(tool, FileSearchTool):
            converted_tool = {
                "type": "file_search",
                "vector_store_ids": tool.vector_store_ids,
            }
            if tool.max_num_results:
                converted_tool["max_num_results"] = tool.max_num_results
            if tool.ranking_options:
                converted_tool["ranking_options"] = tool.ranking_options
            if tool.filters:
                converted_tool["filters"] = tool.filters

            includes = "file_search_call.results" if tool.include_search_results else None
        elif isinstance(tool, ComputerTool):
            converted_tool = {
                "type": "computer_use_preview",
                "environment": tool.computer.environment,
                "display_width": tool.computer.dimensions[0],
                "display_height": tool.computer.dimensions[1],
            }
            includes = None

        else:
            raise UserError(f"Unknown tool type: {type(tool)}, tool")

        return converted_tool, includes

    @classmethod
    def _convert_handoff_tool(cls, handoff: Handoff) -> ToolParam:
        return {
            "name": handoff.tool_name,
            "parameters": handoff.input_json_schema,
            "strict": handoff.strict_json_schema,
            "type": "function",
            "description": handoff.tool_description,
        }
