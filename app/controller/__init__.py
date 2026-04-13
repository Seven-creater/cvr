from .scripted import ScriptedController

__all__ = ["OpenAIResponsesController", "ScriptedController"]


def __getattr__(name: str):
    if name == "OpenAIResponsesController":
        from .openai_responses import OpenAIResponsesController

        return OpenAIResponsesController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
