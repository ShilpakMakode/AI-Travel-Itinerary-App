import os
import sqlite3
import uuid
from datetime import date, datetime
from typing import Dict, List, Tuple

import streamlit as st

from llm_engine import generate_plan, refine_user_answer, run_guardrail


DB_PATH = "navmarg.db"

QUESTIONS: List[Tuple[str, str]] = [
    ("origin", "What is your origin city?"),
    ("destination", "What is your destination city?"),
    ("start_date", "What is your trip start date? (YYYY-MM-DD)"),
    ("end_date", "What is your trip end date? (YYYY-MM-DD)"),
    ("travel_type", "What is your travel type? (Solo/Couple/Family/Friends/Business Trip/Group Tour)"),
    ("adults", "How many adults (13+) are traveling?"),
    ("children", "How many children (0-12) are traveling?"),
    ("budget", "What is your total budget in INR?"),
    ("budget_tier", "What budget tier do you prefer? (Budget/Mid-range/Luxury etc.)"),
    ("interests", "What are your main interests? (comma separated)"),
    ("pace", "What travel pace do you prefer? (Relaxed/Balanced/Active)"),
    ("experience", "What experience style do you want? (Must-see/Hidden Gems/Mix/Local/Instagrammable)"),
]


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            current_question_idx INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS slots (
            session_id TEXT PRIMARY KEY,
            origin TEXT,
            destination TEXT,
            start_date TEXT,
            end_date TEXT,
            travel_type TEXT,
            adults TEXT,
            children TEXT,
            budget TEXT,
            budget_tier TEXT,
            interests TEXT,
            pace TEXT,
            experience TEXT,
            is_complete INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS itineraries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            raw_plan TEXT NOT NULL,
            final_plan TEXT NOT NULL,
            change_request TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def save_session(session_id: str, state: str, current_question_idx: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = _now_iso()
    cur.execute(
        """
        INSERT INTO sessions (id, state, current_question_idx, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            state=excluded.state,
            current_question_idx=excluded.current_question_idx,
            updated_at=excluded.updated_at
        """,
        (session_id, state, current_question_idx, now, now),
    )
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, _now_iso()),
    )
    conn.commit()
    conn.close()


def save_slots(session_id: str, slots: Dict[str, str], is_complete: bool = False) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO slots (
            session_id, origin, destination, start_date, end_date, travel_type,
            adults, children, budget, budget_tier, interests, pace, experience,
            is_complete, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            origin=excluded.origin,
            destination=excluded.destination,
            start_date=excluded.start_date,
            end_date=excluded.end_date,
            travel_type=excluded.travel_type,
            adults=excluded.adults,
            children=excluded.children,
            budget=excluded.budget,
            budget_tier=excluded.budget_tier,
            interests=excluded.interests,
            pace=excluded.pace,
            experience=excluded.experience,
            is_complete=excluded.is_complete,
            updated_at=excluded.updated_at
        """,
        (
            session_id,
            slots.get("origin", ""),
            slots.get("destination", ""),
            slots.get("start_date", ""),
            slots.get("end_date", ""),
            slots.get("travel_type", ""),
            slots.get("adults", ""),
            slots.get("children", ""),
            slots.get("budget", ""),
            slots.get("budget_tier", ""),
            slots.get("interests", ""),
            slots.get("pace", ""),
            slots.get("experience", ""),
            1 if is_complete else 0,
            _now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def save_itinerary(session_id: str, version: int, raw_plan: str, final_plan: str, change_request: str = "") -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO itineraries (session_id, version, raw_plan, final_plan, change_request, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, version, raw_plan, final_plan, change_request, _now_iso()),
    )
    conn.commit()
    conn.close()


def add_assistant_message(content: str) -> None:
    st.session_state.chat_messages.append({"role": "assistant", "content": content})
    save_message(st.session_state.session_id, "assistant", content)


def add_user_message(content: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": content})
    save_message(st.session_state.session_id, "user", content)


def get_current_question() -> Tuple[str, str]:
    idx = st.session_state.current_question_idx
    return QUESTIONS[idx]


def _parse_int(value: str) -> int:
    return int(value.replace(",", "").strip())


def validate_slot(slot: str, value: str) -> Tuple[bool, str]:
    if not value.strip():
        return False, "This answer is empty. Please provide a valid value."

    if slot in {"start_date", "end_date"}:
        try:
            parsed = date.fromisoformat(value.strip())
            if parsed < date.today():
                return False, "Date cannot be in the past. Please use YYYY-MM-DD."
        except ValueError:
            return False, "Invalid date format. Please use YYYY-MM-DD."

    if slot == "end_date":
        start_date = st.session_state.slots.get("start_date", "")
        if start_date:
            try:
                if date.fromisoformat(value.strip()) < date.fromisoformat(start_date):
                    return False, "End date cannot be before start date."
            except ValueError:
                return False, "Invalid end date. Please use YYYY-MM-DD."

    if slot in {"adults", "children", "budget"}:
        try:
            parsed_int = _parse_int(value)
        except ValueError:
            return False, f"Please provide a valid number for {slot}."
        if slot == "adults" and parsed_int < 1:
            return False, "At least 1 adult is required."
        if slot == "children" and parsed_int < 0:
            return False, "Children cannot be negative."
        if slot == "budget" and parsed_int < 1000:
            return False, "Budget should be at least 1000 INR."

    return True, ""


def _format_navmarg_intro() -> str:
    return (
        "Hello, I am **NavMarg**, your AI travel advisor. "
        "I can help you design a complete, personalized itinerary for your trip."
    )


def _ask_next_question() -> None:
    slot, question = get_current_question()
    add_assistant_message(f"{question}\n\n`(Field: {slot})`")


def _generate_initial_itinerary() -> None:
    with st.spinner("NavMarg is creating your itinerary..."):
        raw_plan, final_plan = generate_plan(st.session_state.slots)

    st.session_state.itinerary_version += 1
    st.session_state.latest_raw_plan = raw_plan
    st.session_state.latest_final_plan = final_plan
    save_itinerary(
        st.session_state.session_id,
        st.session_state.itinerary_version,
        raw_plan,
        final_plan,
        "",
    )

    add_assistant_message(final_plan)
    add_assistant_message("Are you satisfied with this plan? Reply `yes` or share changes.")
    st.session_state.agent_state = "awaiting_changes"
    save_session(
        st.session_state.session_id,
        st.session_state.agent_state,
        st.session_state.current_question_idx,
    )


def _generate_refined_itinerary(change_request: str) -> None:
    with st.spinner("NavMarg is refining your itinerary..."):
        raw_plan, final_plan = generate_plan(
            st.session_state.slots,
            previous_itinerary=st.session_state.latest_final_plan,
            change_request=change_request,
        )

    st.session_state.itinerary_version += 1
    st.session_state.latest_raw_plan = raw_plan
    st.session_state.latest_final_plan = final_plan
    save_itinerary(
        st.session_state.session_id,
        st.session_state.itinerary_version,
        raw_plan,
        final_plan,
        change_request,
    )

    add_assistant_message(final_plan)
    add_assistant_message("Updated. Are you satisfied now? Reply `yes` or send more changes.")


def initialize_session() -> None:
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.agent_state = "awaiting_first_user_message"
    st.session_state.onboarding_sent = False
    st.session_state.current_question_idx = 0
    st.session_state.slots = {key: "" for key, _ in QUESTIONS}
    st.session_state.chat_messages = []
    st.session_state.latest_raw_plan = ""
    st.session_state.latest_final_plan = ""
    st.session_state.itinerary_version = 0

    save_session(st.session_state.session_id, st.session_state.agent_state, st.session_state.current_question_idx)


def handle_slot_filling(user_text: str) -> None:
    slot, question = get_current_question()

    try:
        decision = run_guardrail(user_text, expected_slot=slot)
    except Exception:
        decision = {"decision": "ALLOW", "assistant_message": "", "reason": "Guardrail fallback"}

    decision_name = decision.get("decision", "ALLOW").upper()

    if decision_name == "GREETING":
        add_assistant_message(_format_navmarg_intro())
        add_assistant_message(f"Please answer this first:\n\n{question}")
        return

    if decision_name in {"OFFTOPIC", "UNSAFE"}:
        message = decision.get("assistant_message") or "Please share travel-related input so I can continue."
        add_assistant_message(f"{message}\n\nCurrent question: {question}")
        return

    try:
        normalized = refine_user_answer(slot, user_text)
    except Exception:
        normalized = user_text.strip()

    is_valid, error_message = validate_slot(slot, normalized)
    if not is_valid:
        add_assistant_message(f"{error_message}\n\nPlease answer again:\n{question}")
        return

    st.session_state.slots[slot] = normalized.strip()
    save_slots(st.session_state.session_id, st.session_state.slots, is_complete=False)

    st.session_state.current_question_idx += 1
    if st.session_state.current_question_idx < len(QUESTIONS):
        save_session(
            st.session_state.session_id,
            st.session_state.agent_state,
            st.session_state.current_question_idx,
        )
        _ask_next_question()
        return

    save_slots(st.session_state.session_id, st.session_state.slots, is_complete=True)
    _generate_initial_itinerary()


def handle_refinement(user_text: str) -> None:
    lower = user_text.strip().lower()

    if lower in {"yes", "y", "satisfied", "done"}:
        st.session_state.agent_state = "completed"
        save_session(
            st.session_state.session_id,
            st.session_state.agent_state,
            st.session_state.current_question_idx,
        )
        add_assistant_message(
            "Perfect. Glad I could help. If you want a new plan, type `restart`."
        )
        return

    if lower in {"no", "n", "not satisfied"}:
        add_assistant_message("Tell me exactly what to change, and I will regenerate the full itinerary.")
        return

    _generate_refined_itinerary(user_text)


def main() -> None:
    st.set_page_config(page_title="NavMarg Travel Agent", layout="centered")
    init_db()
    initialize_session()

    st.title("NavMarg")
    st.caption("Your AI Travel Agent")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your response...")
    if not user_input:
        return

    add_user_message(user_input)

    if user_input.strip().lower() in {"restart", "start over", "new trip"}:
        st.session_state.initialized = False
        st.rerun()

    if st.session_state.agent_state == "awaiting_first_user_message":
        if not st.session_state.onboarding_sent:
            add_assistant_message(_format_navmarg_intro())
            add_assistant_message(
                "I will ask you a few questions one by one. Once you answer, I will generate your itinerary."
            )
            add_assistant_message("Let's start with your origin city?")
            st.session_state.onboarding_sent = True
            st.session_state.agent_state = "slot_filling"
            save_session(
                st.session_state.session_id,
                st.session_state.agent_state,
                st.session_state.current_question_idx,
            )
    elif st.session_state.agent_state == "slot_filling":
        handle_slot_filling(user_input)
    elif st.session_state.agent_state == "awaiting_changes":
        handle_refinement(user_input)
    else:
        add_assistant_message("Session completed. Type `restart` to begin a new itinerary.")

    st.rerun()


if __name__ == "__main__":
    main()
