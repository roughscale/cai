from __future__ import annotations

from openai import AsyncOpenAI


def is_gpt5_model(model_name: str | None) -> bool:
    if not model_name:
        return False

    normalized = model_name.strip().lower()
    return normalized.startswith("gpt-5")


def should_use_responses_api(
    model_name: str | None,
    use_responses_by_default: bool = False,
) -> bool:
    if is_gpt5_model(model_name):
        return True

    return use_responses_by_default


def create_openai_model_instance(
    *,
    model_name: str,
    openai_client: AsyncOpenAI,
    agent_name: str = "Agent",
    agent_id: str | None = None,
    agent_type: str | None = None,
    use_responses_by_default: bool = False,
):
    from .openai_chatcompletions import OpenAIChatCompletionsModel
    from .openai_responses import OpenAIResponsesModel

    model_cls = (
        OpenAIResponsesModel
        if should_use_responses_api(model_name, use_responses_by_default)
        else OpenAIChatCompletionsModel
    )

    return model_cls(
        model=model_name,
        openai_client=openai_client,
        agent_name=agent_name,
        agent_id=agent_id,
        agent_type=agent_type,
    )
