from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from memory.session_memory import get_history_string, get_chat_messages, add_turn
import agents.rag_agent as rag_agent
import agents.operations_agent as operations_agent

load_dotenv()

_CLASSIFIER_SYSTEM = """You are an intent classifier for a restaurant AI system.

Classify the user message into exactly one category:
- "knowledge" — questions about menu, allergens, opening hours, policies, loyalty rules, events, catering packages, refunds, pet/children policy
- "operations" — requests to check table availability, book a table, get today's special, check loyalty points balance

If the intent is ambiguous, default to "knowledge".

Reply with ONLY one word: "knowledge" or "operations". No punctuation, no explanation."""

_OPS_KEYWORDS = {
    "book", "reserve", "reservation", "availability", "available",
    "table", "special", "today's special", "points", "loyalty points",
    "check my points", "balance", "available times", "avaliable times",
    "yes check",
}
_KNOWLEDGE_KEYWORDS = {
    "menu", "dish", "dishes", "allergen", "allergens", "vegan", "gluten",
    "opening hours", "hours", "policy", "refund", "catering", "events",
    "birthday", "pet", "children", "chicken", "salmon", "pasta",
}

_classifier: ChatGroq | None = None


def _dedupe_response(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if len(blocks) > 1:
        seen = set()
        unique_blocks = []
        for block in blocks:
            key = " ".join(block.split()).casefold()
            if key not in seen:
                seen.add(key)
                unique_blocks.append(block)
        text = "\n\n".join(unique_blocks)

    sentences = [
        part.strip()
        for part in text.split(". ")
        if part.strip()
    ]
    if len(sentences) > 1:
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            key = sentence.rstrip(".").casefold()
            if key not in seen:
                seen.add(key)
                unique_sentences.append(sentence)
        text = ". ".join(unique_sentences)
        if not text.endswith((".", "?", "!")):
            text += "."

    return text


def _get_classifier() -> ChatGroq:
    global _classifier
    if _classifier is None:
        _classifier = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _classifier


def _classify(message: str, history: str) -> str:
    lower_msg = message.lower()
    if any(kw in lower_msg for kw in _OPS_KEYWORDS):
        return "operations"
    if any(kw in lower_msg for kw in _KNOWLEDGE_KEYWORDS):
        return "knowledge"

    context = message
    if history:
        context = f"Conversation so far:\n{history}\n\nLatest message: {message}"

    raw = _get_classifier().invoke([
        SystemMessage(content=_CLASSIFIER_SYSTEM),
        HumanMessage(content=context),
    ]).content.strip().lower()

    if raw in ("knowledge", "operations"):
        return raw

    return "knowledge"


def chat(session_id: str, message: str) -> dict:
    history_str = get_history_string(session_id)
    chat_messages = get_chat_messages(session_id)

    intent = _classify(message, history_str)

    try:
        if intent == "operations":
            response = operations_agent.handle(message, chat_messages)
            agent_used = "operations"
        else:
            response = rag_agent.answer(message, history_str)
            agent_used = "rag"
    except Exception as exc:
        print(f"[orchestrator] sub-agent error: {exc}")
        response = (
            "I'm sorry, I encountered an error processing your request. "
            "Please try again."
        )
        agent_used = intent

    if not response or not response.strip():
        response = (
            "I'm sorry, I couldn't generate a response. "
            "Please try rephrasing your question."
        )
    response = _dedupe_response(response.strip())

    add_turn(session_id, message, response)

    return {"response": response, "agent_used": agent_used}
