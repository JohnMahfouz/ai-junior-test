"""
LangChain @tool wrappers that call the NovaBite MCP Tool Server over HTTP.

The MCP server (tools/mcp_server.py) runs on port 8001 and holds all business
logic and state. These wrappers are what the operations agent sees as "tools".

Start the MCP server first: python tools/mcp_server.py
"""
import os

import requests
from langchain_core.tools import tool

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
_TIMEOUT = 10  # seconds


def _call(endpoint: str, payload: dict) -> str:
    try:
        resp = requests.post(f"{MCP_URL}{endpoint}", json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]
    except requests.ConnectionError:
        return (
            "I'm unable to reach the operations server right now. "
            "Please try again in a moment."
        )
    except requests.HTTPError as exc:
        return f"Operations server returned an error: {exc}"


@tool
def check_table_availability(date: str, time: str, branch: str) -> str:
    """Check whether a table is available at a NovaBite branch.

    Args:
        date: Reservation date in YYYY-MM-DD format, e.g. '2026-05-03'.
        time: Reservation time in HH:MM 24-hour format, e.g. '19:00'.
        branch: Branch name — one of 'downtown', 'uptown', or 'airport'.
    """
    return _call(
        "/tools/check_table_availability",
        {"date": date, "time": time, "branch": branch},
    )


@tool
def list_available_times(date: str, branch: str) -> str:
    """List known available reservation times at a NovaBite branch.

    Args:
        date: Reservation date in YYYY-MM-DD format, e.g. '2026-05-03'.
        branch: Branch name — one of 'downtown', 'uptown', or 'airport'.
    """
    return _call(
        "/tools/list_available_times",
        {"date": date, "branch": branch},
    )


@tool
def list_all_available_slots() -> str:
    """List all known available reservation slots grouped by branch and date."""
    return _call("/tools/list_all_available_slots", {})


@tool
def book_table(name: str, date: str, time: str, branch: str) -> str:
    """Book a table at a NovaBite branch for a guest.

    Args:
        name: Full name for the reservation.
        date: Reservation date in YYYY-MM-DD format, e.g. '2026-05-03'.
        time: Reservation time in HH:MM 24-hour format, e.g. '19:00'.
        branch: Branch name — one of 'downtown', 'uptown', or 'airport'.
    """
    return _call(
        "/tools/book_table",
        {"name": name, "date": date, "time": time, "branch": branch},
    )


@tool
def get_today_special(branch: str) -> str:
    """Get today's special dish at a NovaBite branch.

    Args:
        branch: Branch name — one of 'downtown', 'uptown', or 'airport'.
    """
    return _call("/tools/get_today_special", {"branch": branch})


@tool
def check_loyalty_points(user_id: str) -> str:
    """Check the NovaBite Rewards loyalty points balance for a member.

    Args:
        user_id: The member's loyalty account ID, e.g. 'user_001'.
    """
    return _call("/tools/check_loyalty_points", {"user_id": user_id})


RESTAURANT_TOOLS = [
    check_table_availability,
    list_available_times,
    list_all_available_slots,
    book_table,
    get_today_special,
    check_loyalty_points,
]
