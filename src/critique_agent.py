import streamlit as st
from google import genai
from google.genai import types

def get_gemini_client():
    """
    Safely retrieves the Gemini API Key from Streamlit Secrets 
    and initializes the GenAI Client.
    """
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("🔑 Missing GEMINI_API_KEY in `.streamlit/secrets.toml`!")
        st.stop()
        
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)

def generate_ai_critique(seed_song, candidate_song, top_drivers):
    """
    Sends raw ground-truth feature drivers to Gemini and retrieves a 
    concise, human-centered musical rationale.
    """
    try:
        client = get_gemini_client()
        
        # Format the ground-truth facts extracted from the vector math
        drivers_summary = "\n".join([
            f"- {feat_name}: {int(score * 100)}% similarity match" 
            for feat_name, score in top_drivers
        ])
        
        prompt = f"""
        You are an expert music critic for the VectorTune Recommender Engine.
        Synthesize why '{candidate_song['name']}' by {candidate_song['artists']} 
        was recommended to a listener who likes '{seed_song['name']}' by {seed_song['artists']}.
        
        Ground-Truth Vector Matches (9D Continuous Space):
        {drivers_summary}
        
        Instructions:
        - Write a 2-sentence rationale explaining the acoustic and emotional connection.
        - Mention specific attributes like energy, groove, or acoustic texture based on the matches.
        - Do not output preamble or fluff; return only the 2-sentence musical rationale.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3, # Low temperature prevents hallucination
                max_output_tokens=150
            )
        )
        return response.text.strip()
        
    except Exception as e:
        # Graceful fallback to static text if offline, out of quota, or key error
        return f"Matched based on strong alignment in {top_drivers[0][0]} and {top_drivers[1][0]}."