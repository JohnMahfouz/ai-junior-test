import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.restaurant_tools import (
    RESTAURANT_TOOLS,
    book_table,
    check_loyalty_points,
    check_table_availability,
    get_today_special,
    list_all_available_slots,
    list_available_times,
)

load_dotenv()

_SYSTEM = """You are an operations assistant for NovaBite restaurant chain.
You handle table bookings, availability checks, today's specials, and loyalty point inquiries.

Guidelines:
- Use tools to fulfill every operational request — never guess or invent data.
- If a booking slot is unavailable, suggest checking another time.
- Always confirm booking details (name, date, time, branch) before calling book_table.
- Be friendly, concise, and professional."""

_executor: AgentExecutor | None = None
_BRANCHES = ("downtown", "uptown", "airport")
_AVAILABLE_TIME_PHRASES = (
    "available",
    "available times",
    "avaliable times",
    "availability times",
    "slots",
    "place",
    "places",
    "what times",
    "which times",
    "where is available",
)
_YES_CHECK = {"yes", "yes check", "check", "ok check", "please check"}
_ALL_AVAILABLE_PHRASES = (
    "all available",
    "all the available",
    "all available slots",
    "all the available slots",
    "all available times",
    "all the available times",
    "every branch",
    "every available",
    "where is available",
)


def _build_executor() -> AgentExecutor:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, RESTAURANT_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=RESTAURANT_TOOLS,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )


def _history_text(chat_history: list | None) -> str:
    if not chat_history:
        return ""
    return "\n".join(str(getattr(msg, "content", "")) for msg in chat_history)


def _wants_available_times(message: str, chat_history: list | None) -> bool:
    lower_msg = message.lower().strip()
    if any(phrase in lower_msg for phrase in _AVAILABLE_TIME_PHRASES):
        return True
    if lower_msg in _YES_CHECK:
        return any(
            phrase in _history_text(chat_history).lower()
            for phrase in _AVAILABLE_TIME_PHRASES
        )
    return False


def _wants_all_available_slots(message: str) -> bool:
    lower_msg = message.lower().strip()
    return any(phrase in lower_msg for phrase in _ALL_AVAILABLE_PHRASES)


def _extract_date_and_branch(message: str, chat_history: list | None) -> tuple[str | None, str | None]:
    text = f"{message}\n{_history_text(chat_history)}"
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    branch = next((branch for branch in _BRANCHES if branch in text.lower()), None)
    return (date_match.group(0) if date_match else None, branch)


def _extract_time(message: str) -> str | None:
    match = re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", message)
    return match.group(0) if match else None


def _extract_user_id(message: str) -> str | None:
    match = re.search(r"\buser[_-]\d+\b", message, flags=re.IGNORECASE)
    return match.group(0).replace("-", "_") if match else None


def _extract_name(message: str) -> str | None:
    match = re.search(r"\b(?:for|name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", message)
    return match.group(1) if match else None


def _direct_tool_response(message: str, chat_history: list | None) -> str | None:
    lower_msg = message.lower().strip()

    if _wants_all_available_slots(message):
        return list_all_available_slots.invoke({})

    date, branch = _extract_date_and_branch(message, chat_history)

    if "special" in lower_msg:
        if not branch:
            return "Which branch should I check today's special for?"
        return get_today_special.invoke({"branch": branch})

    if "loyalty" in lower_msg or "points" in lower_msg or "balance" in lower_msg:
        user_id = _extract_user_id(message)
        if not user_id:
            return "Please provide your loyalty user ID."
        return check_loyalty_points.invoke({"user_id": user_id})

    if _wants_available_times(message, chat_history):
        if not date and not branch:
            return list_all_available_slots.invoke({})
        if not date or not branch:
            return (
                "Please provide both the branch and date, or ask for all available "
                "slots to see every branch."
            )
        return list_available_times.invoke({"date": date, "branch": branch})

    if "available" in lower_msg or "availability" in lower_msg:
        time = _extract_time(message)
        if not date or not branch or not time:
            return "Please provide the branch, date, and time to check availability."
        return check_table_availability.invoke({"date": date, "time": time, "branch": branch})

    if "book" in lower_msg or "reserve" in lower_msg:
        time = _extract_time(message) or _extract_time(_history_text(chat_history))
        name = _extract_name(message)
        if not date or not branch or not time:
            return "Please provide the branch, date, and time for the booking."
        if not name:
            return (
                f"To confirm your booking at the {branch.title()} branch on {date} "
                f"at {time}, please provide your full name."
            )
        return book_table.invoke({"name": name, "date": date, "time": time, "branch": branch})

    return None


def handle(message: str, chat_history: list | None = None) -> str:
    direct_response = _direct_tool_response(message, chat_history)
    if direct_response:
        return direct_response

    global _executor
    if _executor is None:
        _executor = _build_executor()

    result = _executor.invoke({
        "input": message,
        "chat_history": chat_history or [],
    })
    return result["output"]
