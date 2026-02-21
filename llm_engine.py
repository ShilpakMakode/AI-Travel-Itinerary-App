import os
from openai import OpenAI
from groq import Groq

def _build_client():
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key), provider

    if provider == "llama":
        # Llama is typically accessed through an OpenAI-compatible API endpoint.
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        if not api_key:
            raise ValueError("LLM_API_KEY (or OPENAI_API_KEY) is not set for llama.")
        if not base_url:
            raise ValueError("LLM_BASE_URL is not set for llama provider.")
        return OpenAI(api_key=api_key, base_url=base_url), provider

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        return Groq(api_key=api_key), provider

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _call_llm(client, provider, model_name, system_prompt, user_prompt):
    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,
    }

    try:
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    if not content or not content.strip():
        raise ValueError("LLM returned an empty response.")

    return content.strip()


def generate_itinerary(data, previous_itinerary=None, refinement_request=None):
    client, provider = _build_client()
    default_model = "llama-3.3-70b-versatile" if provider == "groq" else "gpt-4o-mini"
    model_name = os.getenv("LLM_MODEL", default_model)

    system_prompt = """
You are an expert travel planning AI assistant.

RULES:
- Respect given budget strictly.
- Respect trip duration strictly.
- Return a clear, human-readable itinerary in markdown format.
- Do not return JSON.
- Include daily plans, approximate costs, and practical tips.
- Keep the response concise but useful.
"""

    base_trip_details = f"""
Trip Details:
Origin: {data['origin']}
Destination: {data['destination']}
Start Date: {data['start_date']}
End Date: {data['end_date']}
Days: {data['days']}
Total Budget: {data['budget']}
Budget Tier: {data['budget_tier']}
Travel Type: {data['travel_type']}
Adults: {data['adults']}
Children: {data['children']}
Interests: {data['interests']}
Pace: {data['pace']}
Experience Preference: {data['experience']}
"""

    if refinement_request and previous_itinerary:
        user_prompt = f"""
{base_trip_details}

Previous Itinerary:
{previous_itinerary}

User Customization Request:
{refinement_request}

Create a refined full itinerary (not just delta changes). Keep it practical and budget-consistent.
"""
    else:
        user_prompt = f"""
{base_trip_details}

Generate a complete day-wise itinerary in markdown with this structure:
- Trip summary
- Budget split (stay, food, transport, activities, buffer)
- Day-by-day plan with morning/afternoon/evening
- Estimated cost per day
- Total estimated cost
- Travel tips and safety notes
"""

    return _call_llm(client, provider, model_name, system_prompt, user_prompt)
