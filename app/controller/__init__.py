from .scripted import ScriptedController, ScriptedPolicy, resolve_scripted_policy

__all__ = ["OpenAIResponsesController", "ScriptedController", "ScriptedPolicy", "resolve_scripted_policy"]


def __getattr__(name: str):
    if name == "OpenAIResponsesController":
        from .openai_responses import OpenAIResponsesController

        return OpenAIResponsesController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
