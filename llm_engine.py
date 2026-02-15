import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_itinerary(data):

    system_prompt = """
You are a structured travel planning AI.

RULES:
- Respect given budget.
- Respect trip duration.
- Return ONLY valid JSON.
- Do NOT explain anything.
- Do NOT add text outside JSON.
- Each day must include estimated cost.
- Total estimated cost must not exceed given budget.
"""

    user_prompt = f"""
Trip Details:
Destination: {data['destination']}
Days: {data['days']}
Total Budget: {data['budget']}
Travel Type: {data['travel_type']}
Adults: {data['adults']}
Children: {data['children']}
Interests: {data['interests']}
Pace: {data['pace']}
Experience Preference: {data['experience']}

Generate structured itinerary JSON.
"""

    payload = {
        "model": "llama3.2",
        "prompt": system_prompt + user_prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()

    return result["response"]