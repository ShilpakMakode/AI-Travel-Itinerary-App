import streamlit as st
import requests
from datetime import date
from llm_engine import generate_itinerary

st.set_page_config(page_title="AI Travel Planner", layout="wide")

st.title("AI Travel Itinerary Planner")
st.markdown("Create realistic, validated and budget-aware travel plans.")

# -------------------------
# Session State
# -------------------------
if "itinerary_generated" not in st.session_state:
    st.session_state.itinerary_generated = False

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
            headers={"User-Agent": "travel-app"}
        )
        data = response.json()
        return len(data) > 0
    except:
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

    if not validate_city(origin_city):
        errors.append("Invalid Origin City.")

    if not validate_city(destination_city):
        errors.append("Invalid Destination City.")

    if total_budget <= 0:
        errors.append("Budget must be valid.")

    if travel_type not in ["Solo", "Couple"] and adults + children <= 0:
        errors.append("At least one traveler required.")

    if len(errors) > 0:
        for e in errors:
            st.error(e)
        st.session_state.itinerary_generated = False
    else:
        st.success("Inputs validated successfully.")
        
        
        st.session_state.itinerary_generated = True

        # st.json({
        #     "origin": origin_city,
        #     "destination": destination_city,
        #     "days": total_days,
        #     "travel_type": travel_type,
        #     "adults": adults,
        #     "children": children,
        #     "budget": total_budget,
        #     "tier": budget_tier,
        #     "interests": primary_interests,
        #     "pace": pace,
        #     "experience": experience_type
        
        with st.spinner("Generating AI itinerary..."):
            llm_output = generate_itinerary({
                "destination": destination_city,
                "days": total_days,
                "budget": total_budget,
                "travel_type": travel_type,
                "adults": adults,
                "children": children,
                "interests": primary_interests,
                "pace": pace,
                "experience": experience_type
    })

        st.write(llm_output)
        
        


# =========================
# 6️⃣ FOLLOW-UP CHAT (ONLY AFTER GENERATION)
# =========================

if st.session_state.itinerary_generated:
    st.divider()
    st.subheader("Refine Your Itinerary")

    followup = st.text_input(
        "Modify your plan (e.g., reduce budget, add adventure, remove long drives...)"
    )

    if followup:
        st.write("Refinement request will be processed by AI.")