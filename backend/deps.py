from typing import Optional

_llm = None
_router = None


def get_llm():
    if _llm is None:
        raise RuntimeError("LLM provider not initialized")
    return _llm


def set_llm(llm) -> None:
    global _llm
    _llm = llm


def get_router():
    if _router is None:
        raise RuntimeError("RouterAgent not initialized")
    return _router


def set_router(router) -> None:
    global _router
    _router = router
