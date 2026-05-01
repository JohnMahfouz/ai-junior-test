from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

_MAX_TURNS = 10

_sessions: dict[str, list[BaseMessage]] = {}


def _get_messages(session_id: str) -> list[BaseMessage]:
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


def get_chat_messages(session_id: str) -> list[BaseMessage]:
    return list(_get_messages(session_id))


def get_history_string(session_id: str) -> str:
    messages = _get_messages(session_id)
    if not messages:
        return ""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def add_turn(session_id: str, human: str, ai: str) -> None:
    messages = _get_messages(session_id)
    messages.append(HumanMessage(content=human))
    messages.append(AIMessage(content=ai))
    max_messages = _MAX_TURNS * 2
    if len(messages) > max_messages:
        _sessions[session_id] = messages[-max_messages:]


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
