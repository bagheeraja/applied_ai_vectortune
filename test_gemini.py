# test_gemini.py
import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Load API Key (Check Environment or hardcode temporarily to test)
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    # Try reading from secrets.toml manually
    try:
        import tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomllib.load(f)
            API_KEY = secrets.get("GEMINI_API_KEY")
    except Exception as e:
        print(f"❌ Failed to read secrets.toml: {e}")

if not API_KEY:
    print("❌ GEMINI_API_KEY is missing! Set it or check .streamlit/secrets.toml")
    exit(1)

print(f"🔑 API Key found: {API_KEY[:6]}...{API_KEY[-4:]}")

# 2. Define Pydantic Output Schema
class TrackCritique(BaseModel):
    song_name: str = Field(description="Exact name of the candidate song")
    artist_name: str = Field(description="Artist of the candidate song")
    rationale: str = Field(description="2-sentence musical rationale explaining the vector match")

class RecommendationBatchResponse(BaseModel):
    critiques: list[TrackCritique] = Field(description="List of critiques for all candidate tracks")

# 3. Test API Call
try:
    print("🚀 Dispatching test request to gemini-2.5-flash...")
    client = genai.Client(api_key=API_KEY)
    
    prompt = """
    You are an expert musicologist for the VectorTune Recommender Engine.
    Synthesize why candidate track 'Take On Me' matches seed track 'Blinding Lights'.
    <candidate_song index="1">
        <title>Take On Me</title>
        <artist>a-ha</artist>
        <vector_drivers>danceability: 95% match, energy: 91% match</vector_drivers>
    </candidate_song>
    """

    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=RecommendationBatchResponse,
        )
    )

    print("✅ Success! Raw Gemini Response:")
    print(response.text)
    
    parsed = RecommendationBatchResponse.model_validate_json(response.text)
    print("\n✅ Parsed Pydantic Objects successfully:")
    print(parsed)

except Exception as e:
    print(f"\n❌ GEMINI API CALL FAILED WITH ERROR:\n{type(e).__name__}: {e}")