"""
NovaBite MCP-Style Tool Server

Runs on port 8001 as a standalone service.
Exposes the 4 restaurant tools as real HTTP endpoints so the operations agent
calls an external service over HTTP — just like a real MCP server would.

Start: python tools/mcp_server.py
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="NovaBite MCP Tool Server",
    description="Simulated MCP-style backend exposing restaurant operational tools via HTTP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_availability: dict[tuple, bool] = {
    ("2026-05-03", "18:00", "downtown"): True,
    ("2026-05-03", "19:00", "downtown"): True,
    ("2026-05-03", "20:00", "downtown"): False,
    ("2026-05-03", "18:00", "uptown"): True,
    ("2026-05-03", "19:00", "uptown"): False,
    ("2026-05-04", "18:00", "downtown"): True,
    ("2026-05-04", "19:00", "downtown"): True,
    ("2026-05-04", "18:00", "uptown"): True,
    ("2026-05-04", "20:00", "airport"): True,
    ("2026-05-10", "19:00", "downtown"): True,
    ("2026-05-10", "20:00", "uptown"): True,
}

_bookings: dict[str, dict] = {}

_specials: dict[str, str] = {
    "downtown": "Grilled Sea Bass with lemon-butter caper sauce, served with saffron risotto — $28",
    "uptown": "Slow-roasted lamb shoulder with rosemary jus, roasted root vegetables — $34",
    "airport": "Chicken Tikka Masala with basmati rice and garlic naan — $22",
}

_loyalty_points: dict[str, int] = {
    "user_001": 1250,
    "user_002": 3800,
    "user_003": 500,
    "user_004": 12000,
    "user_005": 200,
}

class AvailabilityRequest(BaseModel):
    date: str
    time: str
    branch: str

class AvailableTimesRequest(BaseModel):
    date: str
    branch: str

class BookingRequest(BaseModel):
    name: str
    date: str
    time: str
    branch: str

class SpecialRequest(BaseModel):
    branch: str

class LoyaltyRequest(BaseModel):
    user_id: str

class ToolResponse(BaseModel):
    message: str

@app.get("/tools", summary="List available tools")
def list_tools():
    return {
        "tools": [
            {
                "name": "check_table_availability",
                "description": "Check if a table is available at a branch on a given date/time",
                "endpoint": "POST /tools/check_table_availability",
            },
            {
                "name": "list_available_times",
                "description": "List known available reservation times at a branch on a given date",
                "endpoint": "POST /tools/list_available_times",
            },
            {
                "name": "list_all_available_slots",
                "description": "List all known available reservation slots grouped by branch and date",
                "endpoint": "POST /tools/list_all_available_slots",
            },
            {
                "name": "book_table",
                "description": "Book a table at a branch for a guest",
                "endpoint": "POST /tools/book_table",
            },
            {
                "name": "get_today_special",
                "description": "Get today's special dish at a branch",
                "endpoint": "POST /tools/get_today_special",
            },
            {
                "name": "check_loyalty_points",
                "description": "Check a member's NovaBite Rewards points balance",
                "endpoint": "POST /tools/check_loyalty_points",
            },
        ]
    }


@app.post("/tools/check_table_availability", response_model=ToolResponse)
def check_table_availability(req: AvailabilityRequest):
    key = (req.date, req.time, req.branch.lower())
    available = _availability.get(key, True)
    if available:
        msg = (
            f"Good news! A table is available at the {req.branch.title()} branch "
            f"on {req.date} at {req.time}."
        )
    else:
        msg = (
            f"Sorry, no tables are available at the {req.branch.title()} branch "
            f"on {req.date} at {req.time}. Please try a different time or date."
        )
    return ToolResponse(message=msg)


@app.post("/tools/list_available_times", response_model=ToolResponse)
def list_available_times(req: AvailableTimesRequest):
    branch = req.branch.lower()
    times = sorted(
        time
        for (date, time, slot_branch), available in _availability.items()
        if date == req.date and slot_branch == branch and available
    )

    if not times:
        return ToolResponse(
            message=(
                f"I don't have any available reservation times listed for "
                f"{req.branch.title()} on {req.date}."
            )
        )

    return ToolResponse(
        message=(
            f"Available times at the {req.branch.title()} branch on {req.date}: "
            f"{', '.join(times)}."
        )
    )


@app.post("/tools/list_all_available_slots", response_model=ToolResponse)
def list_all_available_slots():
    slots: dict[str, dict[str, list[str]]] = {}
    for (date, time, branch), available in sorted(_availability.items()):
        if available:
            slots.setdefault(branch.title(), {}).setdefault(date, []).append(time)

    if not slots:
        return ToolResponse(message="I don't have any available reservation slots listed right now.")

    parts = []
    for branch, dates in slots.items():
        date_parts = [
            f"{date}: {', '.join(sorted(times))}"
            for date, times in sorted(dates.items())
        ]
        parts.append(f"{branch} - {'; '.join(date_parts)}")

    return ToolResponse(message="Available reservation slots: " + " | ".join(parts) + ".")


@app.post("/tools/book_table", response_model=ToolResponse)
def book_table(req: BookingRequest):
    key = (req.date, req.time, req.branch.lower())
    if _availability.get(key, True) is False:
        return ToolResponse(
            message=(
                f"Sorry, no tables are available at {req.branch.title()} on {req.date} "
                f"at {req.time}. Please check availability for another slot first."
            )
        )

    booking_id = f"NB-{1000 + len(_bookings) + 1}"
    _bookings[booking_id] = {
        "name": req.name,
        "date": req.date,
        "time": req.time,
        "branch": req.branch.lower(),
    }
    _availability[key] = False

    return ToolResponse(
        message=(
            f"Booking confirmed! Booking ID: {booking_id} | "
            f"Name: {req.name} | Branch: {req.branch.title()} | {req.date} at {req.time}. "
            "We hold reservations for 15 minutes past the booking time. See you soon!"
        )
    )


@app.post("/tools/get_today_special", response_model=ToolResponse)
def get_today_special(req: SpecialRequest):
    special = _specials.get(req.branch.lower())
    if special:
        return ToolResponse(message=f"Today's special at {req.branch.title()}: {special}")
    return ToolResponse(
        message=(
            f"No special information found for branch '{req.branch}'. "
            "Valid branches: downtown, uptown, airport."
        )
    )


@app.post("/tools/check_loyalty_points", response_model=ToolResponse)
def check_loyalty_points(req: LoyaltyRequest):
    points = _loyalty_points.get(req.user_id)
    if points is None:
        return ToolResponse(
            message=(
                f"No loyalty account found for ID '{req.user_id}'. "
                "Sign up at any branch or on the NovaBite app."
            )
        )

    if points >= 5000:
        redemption = " You qualify for a FREE meal up to $40 value!"
    elif points >= 2000:
        redemption = " You can redeem for a $25 discount."
    elif points >= 1000:
        redemption = " You can redeem for a $12 discount."
    elif points >= 500:
        redemption = " You can redeem for a $5 discount."
    else:
        redemption = f" Earn {500 - points} more points to unlock your first reward."

    return ToolResponse(message=f"Loyalty balance for {req.user_id}: {points:,} points.{redemption}")


@app.get("/health")
def health():
    return {"status": "ok", "service": "NovaBite MCP Tool Server"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
