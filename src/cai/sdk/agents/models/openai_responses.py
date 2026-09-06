from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import litellm
from openai import NOT_GIVEN, APIStatusError, AsyncStream, NotGiven
from openai.types import ChatModel
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
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
from ..tool import ComputerTool, FileSearchTool, FunctionTool, Tool, WebSearchTool
from ..model_trace import write_model_trace
from ..tracing import SpanError, response_span
from ..usage import Usage
from ..version import __version__
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
        agent_name: str = "Agent",
        agent_id: str | None = None,
        agent_type: str | None = None,
    ) -> None:
        self.model = model
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.agent_type = agent_type

        # Track interaction counter and token totals for cli display
        self.interaction_counter = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        
    def set_agent_name(self, name: str) -> None:
        """Set the agent name for CLI display purposes."""
        self.agent_name = name

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
                from cai.util import cli_print_agent_messages
                try:
                    # Create a message-like object to display
                    message_obj = type('ResponseWrapper', (), {
                        'content': '\n'.join([
                            str(item.get('content', '')) if hasattr(item, 'get') 
                            else str(getattr(item, 'text', '')) 
                            for item in response.output 
                            if hasattr(item, 'get') or hasattr(item, 'text')
                        ]),
                        'tool_calls': [
                            type('ToolCallWrapper', (), {
                                'name': item.name,
                                'arguments': item.arguments
                            }) 
                            for item in response.output 
                            if hasattr(item, 'name') and hasattr(item, 'arguments')
                        ]
                    })
                    
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
                        interaction_cost=None,
                        total_cost=None,
                    )
                    
                    # Update token totals
                    self.total_input_tokens += usage.input_tokens
                    self.total_output_tokens += usage.output_tokens
                except Exception as e:
                    logger.error(f"Error printing agent message: {e}")
                
            except Exception as e:
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
                write_model_trace(
                    "model_get_response_error",
                    adapter="openai_responses",
                    agent_name=self.agent_name,
                    model=str(self.model),
                    error=str(e),
                )
                raise

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
        
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
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
                from cai.util import cli_print_agent_messages
                try:
                    # Create a message-like object to display
                    message_obj = type('ResponseWrapper', (), {
                        'content': '\n'.join([
                            str(item.get('content', '')) if hasattr(item, 'get') 
                            else str(getattr(item, 'text', '')) 
                            for item in final_response.output 
                            if hasattr(item, 'get') or hasattr(item, 'text')
                        ]),
                        'tool_calls': [
                            type('ToolCallWrapper', (), {
                                'name': item.name,
                                'arguments': item.arguments
                            }) 
                            for item in final_response.output 
                            if hasattr(item, 'name') and hasattr(item, 'arguments')
                        ]
                    })
                    
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
                        interaction_cost=None,
                        total_cost=None,
                    )
                    
                    # Update token totals
                    self.total_input_tokens += final_response.usage.input_tokens
                    self.total_output_tokens += final_response.usage.output_tokens
                except Exception as e:
                    logger.error(f"Error printing agent message: {e}")
                
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
        list_input = ItemHelpers.input_to_new_input_list(input)

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
                "text": response_format if response_format is not NOT_GIVEN else None,
                "store": model_settings.store,
                "stream": stream,
            },
        )

        kwargs: dict[str, Any] = {
            "model": str(self.model),
            "input": list_input,
            "instructions": system_instructions,
            "include": converted_tools.includes,
            "tools": converted_tools.tools,
            "temperature": model_settings.temperature,
            "top_p": model_settings.top_p,
            "truncation": model_settings.truncation,
            "max_output_tokens": model_settings.max_tokens,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "stream": stream,
            "text": response_format,
            "store": model_settings.store,
            "extra_headers": _HEADERS,
        }
        # litellm does not accept the OpenAI SDK's NOT_GIVEN sentinel
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if v is not None and v is not NOT_GIVEN
        }

        return await litellm.aresponses(**filtered_kwargs)


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
