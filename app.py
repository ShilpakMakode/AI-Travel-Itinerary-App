import os
import re
import sqlite3
import time
import uuid
from datetime import date, datetime
from typing import Dict, List, Tuple

import requests
import streamlit as st

from llm_engine import generate_plan, refine_user_answer, run_guardrail


DB_PATH = "navmarg.db"
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
STORE_RAW_MESSAGES = os.getenv("STORE_RAW_MESSAGES", "false").strip().lower() == "true"
MESSAGE_RETENTION_DAYS = int(os.getenv("MESSAGE_RETENTION_DAYS", "7"))

QUESTIONS: List[Tuple[str, str]] = [
    ("origin", "What is your origin city?"),
    ("destination", "What is your destination city?"),
    ("start_date", "What is your trip start date?"),
    ("end_date", "What is your trip end date?"),
    ("travel_type", "What is your travel type?"),
    ("adults", "How many adults (18+) are traveling?"),
    ("children", "How many children (0-17) are traveling?"),
    ("budget", "What is your total budget in INR?"),
    ("budget_tier", "What budget tier do you prefer?"),
    ("interests", "What are your main interests? Choose all you want."),
    ("pace", "What travel pace do you prefer?"),
    ("experience", "What experience style do you want?"),
]

TRAVEL_TYPES = ["Solo", "Couple", "Family", "Friends", "Business Trip", "Group Tour"]
BUDGET_TIERS = [
    "Ultra Budget (Backpacker)",
    "Budget",
    "Lower Mid-range",
    "Mid-range",
    "Upper Mid-range",
    "Luxury",
    "Ultra Luxury",
]
PACE_OPTIONS = ["Very Relaxed", "Relaxed", "Balanced", "Active", "Fast-paced"]
EXPERIENCE_OPTIONS = [
    "Must-see Landmarks",
    "Mostly Hidden Gems",
    "Mix of Both",
    "Highly Local Experiences",
    "Instagrammable Spots",
]
INTEREST_OPTIONS = [
    "Nature & Landscapes",
    "Mountains",
    "Beaches",
    "Wildlife",
    "Adventure Sports",
    "Road Trips",
    "History & Heritage",
    "Food & Culinary",
    "Nightlife",
    "Shopping",
    "Spiritual",
    "Photography",
    "Luxury Experiences",
    "Wellness & Spa",
    "Local Culture",
    "Festivals",
]


def validate_city(city: str) -> Tuple[bool, bool]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    for _ in range(2):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": "navmarg-travel-agent"},
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
            return len(data) > 0, False
        except requests.RequestException:
            time.sleep(0.3)
    return False, True


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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS guardrail_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            slot TEXT,
            decision TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        DELETE FROM messages
        WHERE julianday(created_at) < julianday('now', ?)
        """,
        (f"-{MESSAGE_RETENTION_DAYS} days",),
    )
    conn.commit()
    conn.close()


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _sanitize_for_storage(content: str) -> str:
    text = content or ""
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[redacted-email]", text)
    text = re.sub(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b", "[redacted-phone]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:240]


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
    stored_content = content if STORE_RAW_MESSAGES else _sanitize_for_storage(content)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, stored_content, _now_iso()),
    )
    conn.commit()
    conn.close()


def log_guardrail_decision(session_id: str, slot: str, decision: str, reason: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO guardrail_audit (session_id, slot, decision, reason, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, slot, decision, reason, _now_iso()),
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


def add_assistant_message(content: str, stream: bool = False) -> None:
    cleaned = content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    st.session_state.chat_messages.append(
        {"id": str(uuid.uuid4()), "role": "assistant", "content": cleaned, "stream": stream}
    )
    save_message(st.session_state.session_id, "assistant", content)


def add_user_message(content: str) -> None:
    st.session_state.chat_messages.append({"id": str(uuid.uuid4()), "role": "user", "content": content})
    save_message(st.session_state.session_id, "user", content)


def get_current_question() -> Tuple[str, str]:
    idx = st.session_state.current_question_idx
    return QUESTIONS[idx]


def _is_question_index_exhausted() -> bool:
    return st.session_state.current_question_idx >= len(QUESTIONS)


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
        if slot == "adults" and parsed_int > 15:
            return False, "Adults cannot be more than 15."
        if slot == "children" and parsed_int < 0:
            return False, "Children cannot be negative."
        if slot == "children" and parsed_int > 15:
            return False, "Children cannot be more than 15."
        if slot == "budget" and parsed_int < 1000:
            return False, "Budget should be at least 1000 INR."
        if slot == "budget" and parsed_int > 5000000:
            return False, "Please enter a realistic budget (up to 50,00,000 INR)."

    if slot == "origin":
        is_valid, unavailable = validate_city(value.strip())
        if unavailable:
            return False, "City verification is temporarily unavailable. Please recheck spelling and try again."
        if not is_valid:
            return False, "Origin location looks invalid. Please enter a proper city name."

    if slot == "destination":
        is_valid, unavailable = validate_city(value.strip())
        if unavailable:
            return False, "City verification is temporarily unavailable. Please recheck spelling and try again."
        if not is_valid:
            return False, "Destination location looks invalid. Please enter a proper city name."

    return True, ""


def _format_navmarg_intro() -> str:
    return (
        "Hello, I am **NavMarg**, your AI travel advisor. "
        "I can help you design a complete, personalized itinerary for your trip."
    )


def _ask_next_question() -> None:
    _, question = get_current_question()
    add_assistant_message(question)


def _auto_apply_travel_type_defaults_and_skip() -> bool:
    travel_type = st.session_state.slots.get("travel_type", "").strip().lower()
    if travel_type not in {"solo", "couple"}:
        return False

    if travel_type == "solo":
        st.session_state.slots["adults"] = "1"
        st.session_state.slots["children"] = "0"
    else:
        st.session_state.slots["adults"] = "2"
        st.session_state.slots["children"] = "0"

    idx = st.session_state.current_question_idx
    changed = False
    while idx < len(QUESTIONS) and QUESTIONS[idx][0] in {"adults", "children"}:
        idx += 1
        changed = True

    if changed:
        st.session_state.current_question_idx = idx
        save_slots(st.session_state.session_id, st.session_state.slots, is_complete=False)
    return changed


def _word_stream(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


def _generate_initial_itinerary() -> None:
    with st.spinner("NavMarg is creating your itinerary..."):
        try:
            raw_plan, final_plan = generate_plan(st.session_state.slots)
        except Exception as exc:
            err = str(exc)
            if "GROQ_API_KEY" in err:
                msg = (
                    f"Could not generate itinerary right now: {err}\n\n"
                    "Please set `GROQ_API_KEY` and try again."
                )
            else:
                msg = f"Could not generate itinerary right now: {err}\n\nPlease try again."
            add_assistant_message(msg)
            return

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

    add_assistant_message(final_plan, stream=True)
    add_assistant_message("Are you satisfied with this plan? Reply `yes` or share changes.")
    st.session_state.agent_state = "awaiting_changes"
    save_session(
        st.session_state.session_id,
        st.session_state.agent_state,
        st.session_state.current_question_idx,
    )


def _generate_refined_itinerary(change_request: str) -> None:
    with st.spinner("NavMarg is refining your itinerary..."):
        try:
            raw_plan, final_plan = generate_plan(
                st.session_state.slots,
                previous_itinerary=st.session_state.latest_final_plan,
                change_request=change_request,
            )
        except Exception as exc:
            err = str(exc)
            if "GROQ_API_KEY" in err:
                msg = (
                    f"Could not refine itinerary right now: {err}\n\n"
                    "Please verify `GROQ_API_KEY` and try again."
                )
            else:
                msg = f"Could not refine itinerary right now: {err}\n\nPlease try again."
            add_assistant_message(msg)
            return

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

    add_assistant_message(final_plan, stream=True)
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
    st.session_state.last_activity_ts = time.time()

    save_session(st.session_state.session_id, st.session_state.agent_state, st.session_state.current_question_idx)


def _store_current_slot_answer(user_text: str) -> None:
    slot, question = get_current_question()

    structured_slots = {
        "start_date",
        "end_date",
        "travel_type",
        "adults",
        "children",
        "budget",
        "budget_tier",
        "pace",
        "experience",
    }

    if slot not in structured_slots:
        try:
            decision = run_guardrail(user_text, expected_slot=slot)
        except Exception:
            decision = {"decision": "ALLOW", "assistant_message": "", "reason": "Guardrail fallback"}

        decision_name = decision.get("decision", "ALLOW").upper()
        log_guardrail_decision(
            st.session_state.session_id,
            slot,
            decision_name,
            decision.get("reason", ""),
        )
        if decision_name == "GREETING":
            add_assistant_message(_format_navmarg_intro(), stream=True)
            add_assistant_message(f"Please answer this first:\n\n{question}", stream=True)
            return

        if decision_name in {"OFFTOPIC", "UNSAFE"}:
            message = decision.get("assistant_message") or "Please share travel-related input so I can continue."
            add_assistant_message(f"{message}\n\nCurrent question: {question}")
            return

        try:
            normalized = refine_user_answer(slot, user_text)
        except Exception:
            normalized = user_text.strip()
    else:
        normalized = user_text.strip()

    is_valid, error_message = validate_slot(slot, normalized)
    if not is_valid:
        add_assistant_message(f"{error_message}\n\nPlease answer again:\n{question}")
        return

    if slot in {"adults", "children", "budget"}:
        st.session_state.slots[slot] = str(_parse_int(normalized))
    else:
        st.session_state.slots[slot] = normalized.strip()
    save_slots(st.session_state.session_id, st.session_state.slots, is_complete=False)

    st.session_state.current_question_idx += 1
    if slot == "travel_type":
        skipped = _auto_apply_travel_type_defaults_and_skip()
        if skipped:
            add_assistant_message(
                "Since you selected Solo/Couple, I have auto-set traveler counts and skipped adults/children questions."
            )

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
    if len(user_text.strip()) > 800:
        add_assistant_message(
            "Your change request is too long. Please keep it under 800 characters for reliable processing."
        )
        return

    blocked_keywords = [
        "gun",
        "weapon",
        "buy gun",
        "sex",
        "s*x",
        "porn",
        "escort",
        "drugs",
        "cocaine",
        "heroin",
        "kill",
        "murder",
    ]
    if any(keyword in lower for keyword in blocked_keywords):
        add_assistant_message(
            "I can only help with safe travel planning. "
            "Please avoid requests related to weapons, sexual content, drugs, or violence."
        )
        log_guardrail_decision(
            st.session_state.session_id,
            "refinement",
            "UNSAFE",
            "Blocked keyword match in refinement request",
        )
        return

    try:
        decision = run_guardrail(user_text, expected_slot="refinement")
    except Exception:
        decision = {"decision": "ALLOW", "assistant_message": "", "reason": "Guardrail fallback"}

    decision_name = decision.get("decision", "ALLOW").upper()
    log_guardrail_decision(
        st.session_state.session_id,
        "refinement",
        decision_name,
        decision.get("reason", ""),
    )
    if decision_name in {"UNSAFE", "OFFTOPIC"}:
        add_assistant_message(
            "Please keep your request strictly related to travel itinerary improvements."
        )
        return

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


def _render_slot_input_widget() -> None:
    if _is_question_index_exhausted():
        st.markdown("---")
        st.info("All required details are collected. Click below to generate itinerary.")
        if st.button("Generate Itinerary"):
            _generate_initial_itinerary()
            st.rerun()
        return

    if _auto_apply_travel_type_defaults_and_skip():
        if st.session_state.current_question_idx >= len(QUESTIONS):
            save_slots(st.session_state.session_id, st.session_state.slots, is_complete=True)
            _generate_initial_itinerary()
        else:
            _ask_next_question()
        st.rerun()

    slot, question = get_current_question()
    st.markdown("---")

    with st.form(f"slot_form_{slot}"):
        submitted = False
        answer_text = ""
        display_text = ""

        if slot in {"origin", "destination"}:
            val = st.text_input("Your answer")
            submitted = st.form_submit_button("Submit")
            answer_text = val.strip()
            display_text = answer_text
        elif slot in {"start_date", "end_date"}:
            min_date = date.today()
            if slot == "end_date" and st.session_state.slots.get("start_date"):
                min_date = date.fromisoformat(st.session_state.slots["start_date"])
            val = st.date_input("Select date", min_value=min_date)
            submitted = st.form_submit_button("Submit")
            answer_text = val.isoformat()
            display_text = answer_text
        elif slot == "travel_type":
            val = st.selectbox("Select travel type", TRAVEL_TYPES)
            submitted = st.form_submit_button("Submit")
            answer_text = val
            display_text = val
        elif slot == "adults":
            val = st.number_input("Number of adults (18+)", min_value=1, max_value=15, step=1)
            submitted = st.form_submit_button("Submit")
            answer_text = str(int(val))
            display_text = answer_text
        elif slot == "children":
            val = st.number_input("Number of children (0-17)", min_value=0, max_value=15, step=1)
            submitted = st.form_submit_button("Submit")
            answer_text = str(int(val))
            display_text = answer_text
        elif slot == "budget":
            val = st.number_input(
                "Total budget (INR)",
                min_value=1000,
                max_value=5000000,
                step=1000,
                format="%d",
            )
            submitted = st.form_submit_button("Submit")
            answer_text = str(int(val))
            display_text = f"{int(val):,}"
        elif slot == "budget_tier":
            val = st.selectbox("Select budget tier", BUDGET_TIERS)
            submitted = st.form_submit_button("Submit")
            answer_text = val
            display_text = val
        elif slot == "interests":
            val = st.multiselect("Choose all interests you want", INTEREST_OPTIONS)
            submitted = st.form_submit_button("Submit")
            answer_text = ", ".join(val)
            display_text = answer_text
        elif slot == "pace":
            val = st.selectbox("Select pace", PACE_OPTIONS)
            submitted = st.form_submit_button("Submit")
            answer_text = val
            display_text = val
        elif slot == "experience":
            val = st.selectbox("Select experience style", EXPERIENCE_OPTIONS)
            submitted = st.form_submit_button("Submit")
            answer_text = val
            display_text = val

        if submitted:
            if not answer_text:
                add_assistant_message(f"Please answer the question first.\n\n{question}")
                st.rerun()

            add_user_message(display_text)
            st.session_state.last_activity_ts = time.time()
            _store_current_slot_answer(answer_text)
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="NavMarg Travel Agent", layout="centered")
    init_db()
    initialize_session()

    st.title("NavMarg")
    st.caption("Your AI Travel Agent")
    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "Missing `GROQ_API_KEY`. Set it before generating itineraries.\n"
            "Example: `export GROQ_API_KEY=\"your_key\"`"
        )
    st.caption(
        f"Session timeout: {SESSION_TIMEOUT_MINUTES} min | "
        f"Max output tokens: {os.getenv('MAX_OUTPUT_TOKENS', '1800')}"
    )

    now_ts = time.time()
    last_ts = st.session_state.get("last_activity_ts", now_ts)
    if st.session_state.get("current_question_idx", 0) > len(QUESTIONS):
        st.session_state.current_question_idx = len(QUESTIONS)
    if now_ts - last_ts > SESSION_TIMEOUT_MINUTES * 60:
        st.warning("Session expired due to inactivity. Starting a fresh session.")
        st.session_state.initialized = False
        st.rerun()

    for idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("stream"):
                st.write_stream(_word_stream(msg["content"]))
                st.session_state.chat_messages[idx]["stream"] = False
            else:
                st.markdown(msg["content"])

    if st.session_state.agent_state == "slot_filling":
        _render_slot_input_widget()
        return

    user_input = st.chat_input("Type your response...")
    if not user_input:
        return

    add_user_message(user_input)
    st.session_state.last_activity_ts = time.time()
    cleaned_input = user_input.strip().lower()

    if cleaned_input in {"restart", "start over", "new trip"}:
        st.session_state.initialized = False
        st.rerun()

    if st.session_state.agent_state == "awaiting_first_user_message":
        if not st.session_state.onboarding_sent:
            add_assistant_message(_format_navmarg_intro(), stream=True)
            add_assistant_message(
                "I will ask you a few questions one by one. Once you answer, I will generate your itinerary.",
                stream=True,
            )
            add_assistant_message("Let's start with your origin city?", stream=True)
            st.session_state.onboarding_sent = True
            st.session_state.agent_state = "slot_filling"
            _ask_next_question()
            save_session(
                st.session_state.session_id,
                st.session_state.agent_state,
                st.session_state.current_question_idx,
            )
    elif st.session_state.agent_state == "awaiting_changes":
        handle_refinement(user_input)
    else:
        add_assistant_message("Session completed. Type `restart` to begin a new itinerary.")

    st.rerun()


if __name__ == "__main__":
    main()
    