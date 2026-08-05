
import os
import sys
import json
import numpy as np
from tabulate import tabulate

# -------------------------------------------------------------------
# Dynamically add the directory containing main.py to Python's sys.path
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Relative imports for files in the same directory
from data_loader import initialize_engine
from recommender import apply_scoring_rule, apply_ranking_rule, generate_feature_differentials
from critique_agent import generate_batch_ai_critiques


def build_seed_vector(prefs, feature_cols):
    """
    Builds a pure 9D seed vector from a preferences dictionary matching `feature_cols`.
    """
    return np.array([prefs[col] for col in feature_cols], dtype=np.float32)


# Distinct user preference profiles expressed across the 9 continuous feature dimensions
# (0-1 range for all normalized features).
STRESS_TEST_PROFILES = {
    "High-Energy Pop": {
        "danceability": 0.80, "energy": 0.90, "speechiness": 0.06,
        "acousticness": 0.05, "instrumentalness": 0.00, "liveness": 0.15,
        "valence": 0.85, "tempo_norm": 0.60, "loudness_norm": 0.85,
    },
    "Chill Lofi": {
        "danceability": 0.50, "energy": 0.25, "speechiness": 0.04,
        "acousticness": 0.85, "instrumentalness": 0.40, "liveness": 0.10,
        "valence": 0.40, "tempo_norm": 0.30, "loudness_norm": 0.30,
    },
    "Deep Intense Rock": {
        "danceability": 0.35, "energy": 0.95, "speechiness": 0.08,
        "acousticness": 0.02, "instrumentalness": 0.05, "liveness": 0.35,
        "valence": 0.30, "tempo_norm": 0.75, "loudness_norm": 0.95,
    },
    # Adversarial / edge-case profiles
    "Conflicting: High Energy + Sad": {
        "danceability": 0.30, "energy": 0.90, "speechiness": 0.05,
        "acousticness": 0.10, "instrumentalness": 0.10, "liveness": 0.20,
        "valence": 0.05, "tempo_norm": 0.70, "loudness_norm": 0.80,
    },
    "All Extremes (min/max mix)": {
        "danceability": 1.00, "energy": 0.00, "speechiness": 0.95,
        "acousticness": 1.00, "instrumentalness": 1.00, "liveness": 1.00,
        "valence": 0.00, "tempo_norm": 0.00, "loudness_norm": 1.00,
    },
    "Flatline (all neutral 0.5)": {
        "danceability": 0.50, "energy": 0.50, "speechiness": 0.50,
        "acousticness": 0.50, "instrumentalness": 0.50, "liveness": 0.50,
        "valence": 0.50, "tempo_norm": 0.50, "loudness_norm": 0.50,
    },
}


def run_stress_test():
    """Runs the recommender against STRESS_TEST_PROFILES and prints top-5 results."""
    df, feature_matrix, feature_cols = initialize_engine()

    for name, prefs in STRESS_TEST_PROFILES.items():
        seed_vector = build_seed_vector(prefs, feature_cols)
        scores = apply_scoring_rule(seed_vector, feature_matrix)
        
        # FIXED: Pass only positional arguments matching updated apply_ranking_rule signature
        recommendations = apply_ranking_rule(scores, df, top_n=5)

        print("=" * 80)
        print(f"Profile: {name}")
        print(f"  {prefs}")
        print("-" * 80)
        print(tabulate(recommendations, headers="keys", tablefmt="fancy_grid", showindex=False))
        print()


def run_app():
    import streamlit as st

    st.set_page_config(page_title="Music Vector Engine", layout="centered")

    st.title("🎵 VectorTune")

    # Initialize and pull shared data structures from cache
    with st.spinner("Caching and normalizing 1.2M track data matrix..."):
        df, feature_matrix, feature_cols = initialize_engine()

    st.write(f"Engine online. Currently indexing **{len(df):,}** clean tracks in RAM.")

    # Search UI inputs
    search_song = st.text_input("Enter a Song Name:", placeholder="e.g., Blinding Lights")
    search_artist = st.text_input("Enter Artist Name (Optional):", placeholder="e.g., The Weeknd")

    if st.button("Generate Recommendations", type="primary"):
        if search_song:
            # String comparison mask building
            query_mask = df['name'].str.lower() == search_song.lower()
            if search_artist:
                query_mask = query_mask & (df['artists'].str.lower().str.contains(search_artist.lower()))

            song_lookup = df[query_mask]

            if not song_lookup.empty:
                matched_song = song_lookup.iloc[0]
                st.success(f"Target track found: **{matched_song['name']}** by *{matched_song['artists']}*")

                # Extract target vector coordinates
                seed_vector = matched_song[feature_cols].to_numpy(dtype=np.float32)

                # Execute calculation pipelines
                with st.spinner("Processing linear algebra matching matrix..."):
                    scores = apply_scoring_rule(seed_vector, feature_matrix)
                    
                    # FIXED: Pass only supported parameters
                    recommendations = apply_ranking_rule(scores, df, top_n=10)

                # Deduplicate matching tracks (remove the seed song from its own recommendations list)
                recommendations = recommendations[
                    ~((recommendations['name'].str.lower() == matched_song['name'].lower()) &
                      (recommendations['artists'].str.lower() == matched_song['artists'].lower()))
                ].head(5)

                # -------------------------------------------------------------------
                # Prepare JSON batch payload for single-pass Gemini API call
                # -------------------------------------------------------------------
                candidates_payload = []
                for idx, row in recommendations.iterrows():
                    candidate_vector = df.loc[row.name, feature_cols].to_numpy(dtype=np.float32)
                    top_drivers, _ = generate_feature_differentials(seed_vector, candidate_vector, feature_cols)
                    
                    candidates_payload.append({
                        "name": str(row['name']),
                        "artists": str(row['artists']),
                        "drivers": [(str(k), float(v)) for k, v in top_drivers]
                    })

                candidates_json_str = json.dumps(candidates_payload)

                # Execute single batch critique request
                with st.spinner("Gemini AI Agent is synthesizing musical rationales..."):
                    critiques_map = generate_batch_ai_critiques(
                        seed_name=str(matched_song['name']),
                        seed_artist=str(matched_song['artists']),
                        candidates_json_str=candidates_json_str
                    )

                # -------------------------------------------------------------------
                # Render Recommendations Feed UI
                # -------------------------------------------------------------------
                st.subheader("🎵 Your Custom Algorithmic Matches:")
                for idx, row in recommendations.iterrows():
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {row['name']}")
                            st.markdown(f"*{row['artists']}*")
                            
                            # Retrieve cached rationale from dictionary with normalized key
                            song_key = str(row['name']).strip().lower()
                            default_fallback = f"Matched based on alignment in key continuous vector features."
                            rationale = critiques_map.get(song_key, default_fallback)
                            
                            st.markdown("**AI Agent Rationale:**")
                            st.info(rationale)
                            
                        with col2:
                            st.metric(label="Match Score", value=f"{row['match_score']}%")
            else:
                st.error("Song not found in the local database catalog. Please verify spelling.")
        else:
            st.warning("Please enter at least a track name to initialize recommendations processing.")


if __name__ == "__main__":
    if "--stress-test" in sys.argv:
        run_stress_test()
    else:
        run_app()