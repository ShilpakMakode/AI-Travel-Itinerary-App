import streamlit as st
import requests
import os
from datetime import date
from llm_engine import generate_itinerary

st.set_page_config(page_title="AI Travel Planner", layout="wide")

st.title("AI Travel Itinerary Planner")
st.markdown("Create realistic, validated and budget-aware travel plans.")
st.caption(
    f"LLM Provider: `{os.getenv('LLM_PROVIDER', 'openai')}` | "
    f"Model: `{os.getenv('LLM_MODEL', 'default')}`"
)

# -------------------------
# Session State
# -------------------------
if "itinerary_generated" not in st.session_state:
    st.session_state.itinerary_generated = False
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "trip_payload" not in st.session_state:
    st.session_state.trip_payload = None

# -------------------------
# Utility: Validate City
# -------------------------
def validate_city(city):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city,
        "format": "json",
        "limit": 1
    }
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "travel-app"},
            timeout=8
        )
        response.raise_for_status()
        data = response.json()
        return len(data) > 0
    except requests.RequestException:
        return False


# =========================
# 1️⃣ CORE LOGISTICS
# =========================

st.subheader("1. Core Logistics")

col1, col2 = st.columns(2)

with col1:
    origin_city = st.text_input("Origin City")
    destination_city = st.text_input("Destination City")

with col2:
    start_date = st.date_input("From Date", min_value=date.today())
    end_date = st.date_input("To Date", min_value=start_date)

# ✅ REAL-TIME DAY CALCULATION (Now Works)
total_days = (end_date - start_date).days + 1
st.info(f"Total Trip Duration: **{total_days} days**")

travel_type = st.selectbox(
    "Travel Type",
    [
        "Solo",
        "Couple",
        "Family",
        "Friends",
        "Business Trip",
        "Group Tour"
    ]
)

# =========================
# 2️⃣ DYNAMIC TRAVELER INPUTS
# =========================

adults = 1
children = 0

if travel_type not in ["Solo", "Couple"]:
    st.markdown("### Group Details")
    adults = st.number_input(
        "Number of Adults (Age 13 and above)",
        min_value=1,
        step=1
    )
    children = st.number_input(
        "Number of Children (Age 0–12)",
        min_value=0,
        step=1
    )

# =========================
# 3️⃣ BUDGET
# =========================

st.subheader("2. Budget Constraints")

total_budget = st.number_input(
    "Total Budget (INR)",
    min_value=1000,
    max_value=100000000,
    step=1000,
    format="%d"
)

budget_tier = st.selectbox(
    "Budget Tier",
    [
        "Ultra Budget (Backpacker)",
        "Budget",
        "Lower Mid-range",
        "Mid-range",
        "Upper Mid-range",
        "Luxury",
        "Ultra Luxury"
    ]
)

# =========================
# 4️⃣ TRAVEL STYLE
# =========================

st.subheader("3. Travel Style")

primary_interests = st.multiselect(
    "Primary Interests",
    [
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
        "Festivals"
    ]
)

pace = st.selectbox(
    "Pace of Travel",
    [
        "Very Relaxed",
        "Relaxed",
        "Balanced",
        "Active",
        "Fast-paced"
    ]
)

experience_type = st.selectbox(
    "Experience Preference",
    [
        "Must-see Landmarks",
        "Mostly Hidden Gems",
        "Mix of Both",
        "Highly Local Experiences",
        "Instagrammable Spots"
    ]
)

# =========================
# 5️⃣ GENERATE BUTTON
# =========================

if st.button("Generate Itinerary"):

    errors = []

    if not origin_city or not destination_city:
        errors.append("Origin and Destination cities are required.")

    if origin_city and not validate_city(origin_city):
        errors.append("Invalid Origin City.")

    if destination_city and not validate_city(destination_city):
        errors.append("Invalid Destination City.")

    if total_budget <= 0:
        errors.append("Budget must be valid.")

    if travel_type not in ["Solo", "Couple"] and adults + children <= 0:
        errors.append("At least one traveler required.")

    if len(errors) > 0:
        for e in errors:
            st.error(e)
        st.session_state.itinerary_generated = False
        st.session_state.itinerary = None
    else:
        st.success("Inputs validated successfully.")

        payload = {
            "origin": origin_city,
            "destination": destination_city,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": total_days,
            "budget": int(total_budget),
            "budget_tier": budget_tier,
            "travel_type": travel_type,
            "adults": int(adults),
            "children": int(children),
            "interests": primary_interests,
            "pace": pace,
            "experience": experience_type
        }

        with st.spinner("Generating AI itinerary..."):
            try:
                llm_output = generate_itinerary(payload)
                st.session_state.itinerary = llm_output
                st.session_state.trip_payload = payload
                st.session_state.itinerary_generated = True
            except Exception as exc:
                st.session_state.itinerary_generated = False
                st.session_state.itinerary = None
                st.session_state.trip_payload = None
                st.error(f"Failed to generate itinerary: {exc}")
                st.info(
                    "If using Groq, set GROQ_API_KEY and LLM_PROVIDER=groq. "
                    "If you see rate/quota errors, check your Groq usage limits."
                )

if st.session_state.itinerary:
    st.subheader("Generated Itinerary")
    st.markdown(st.session_state.itinerary)


# =========================
# 6️⃣ FOLLOW-UP CHAT (ONLY AFTER GENERATION)
# =========================

if st.session_state.itinerary_generated:
    st.divider()
    st.subheader("Refine Your Itinerary")

    with st.form("refine_itinerary_form"):
        followup = st.text_input(
            "Modify your plan (e.g., reduce budget, add adventure, remove long drives...)"
        )
        refine_submitted = st.form_submit_button("Refine Itinerary")

    if refine_submitted:
        if not followup.strip():
            st.error("Please enter a customization request.")
        elif not st.session_state.trip_payload or not st.session_state.itinerary:
            st.error("Missing itinerary context. Please generate itinerary again.")
        else:
            with st.spinner("Refining itinerary..."):
                try:
                    refined_output = generate_itinerary(
                        st.session_state.trip_payload,
                        previous_itinerary=st.session_state.itinerary,
                        refinement_request=followup.strip()
                    )
                    st.session_state.itinerary = refined_output
                    st.success("Itinerary refined.")
                except Exception as exc:
                    st.error(f"Failed to refine itinerary: {exc}")
