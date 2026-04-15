def init_memory(session_state):
    if "messages" not in session_state:
        session_state.messages = []


def add_message(session_state, role: str, content, route: str = "chat", tool: str = None, docs=None, extra=None):
    session_state.messages.append(
        {
            "role": role,
            "content": content,
            "route": route,
            "tool": tool,
            "docs": docs or [],
            "extra": extra or [],
        }
    )


def clear_memory(session_state):
    session_state.messages = []


def get_history(session_state, limit: int = 6):
    return session_state.messages[-limit:]


def format_history_for_prompt(history, limit: int = 6) -> str:
    if not history:
        return ""

    selected = history[-limit:]
    lines = []

    for msg in selected:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role} : {msg['content']}")

    return "\n".join(lines)