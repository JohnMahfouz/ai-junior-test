from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from rag.retriever import get_retriever
from prompts.rag_prompt import rag_prompt

load_dotenv()

_chain = None
_MENU_PATH = Path(__file__).parent.parent / "data" / "menu.md"
_FULL_MENU_TRIGGERS = (
    "all the menu",
    "full menu",
    "entire menu",
    "whole menu",
    "see the menu",
    "show me the menu",
    "view the menu",
)


def _format_docs(docs: list) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def _build_chain():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = get_retriever(k=4)

    return (
        {
            "context": itemgetter("retrieval_query") | retriever | RunnableLambda(_format_docs),
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )


def _is_full_menu_request(question: str) -> bool:
    normalized = question.lower()
    return any(trigger in normalized for trigger in _FULL_MENU_TRIGGERS)


def _format_full_menu() -> str:
    menu_text = _MENU_PATH.read_text(encoding="utf-8")
    lines = []
    current_section = None

    for raw_line in menu_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## ") and not line.startswith("### "):
            current_section = line.removeprefix("## ").strip()
            if current_section != "Allergen Summary":
                lines.append(f"\n{current_section}:")
            continue
        if line.startswith("### ") and current_section != "Allergen Summary":
            item = line.removeprefix("### ").strip()
            lines.append(f"- {item}")
            continue
        if line.startswith("- Price:") and lines:
            lines[-1] = f"{lines[-1]} ({line.removeprefix('- Price:').strip()})"

    return (
        "Here is the NovaBite menu:\n"
        f"{chr(10).join(lines).strip()}\n\n"
        "[Source: menu.md]"
    )


_rephrase_llm: ChatGroq | None = None

_REPHRASE_PROMPT = """Given the conversation history below, rephrase the follow-up question
into a short, self-contained search query that can be used to find the answer
in a restaurant knowledge base. If the question is already standalone, return it as-is.
Do NOT answer the question — only rephrase it.

Conversation history:
{history}

Follow-up question: {question}

Standalone search query:"""


def _rephrase_query(question: str, history: str) -> str:
    """Use a fast LLM call to turn a follow-up into a standalone retrieval query."""
    global _rephrase_llm
    if _rephrase_llm is None:
        _rephrase_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    result = _rephrase_llm.invoke(
        _REPHRASE_PROMPT.format(history=history, question=question)
    )
    return result.content.strip()


def answer(question: str, conversation_history: str = "") -> str:
    global _chain
    if _is_full_menu_request(question):
        return _format_full_menu()

    if _chain is None:
        _chain = _build_chain()

    if conversation_history:
        retrieval_query = _rephrase_query(question, conversation_history)
        llm_question = (
            f"Previous conversation:\n{conversation_history}\n\nCurrent question: {question}"
        )
    else:
        retrieval_query = question
        llm_question = question

    return _chain.invoke({
        "retrieval_query": retrieval_query,
        "question": llm_question,
    })
