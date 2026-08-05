# src/recommender.py
import numpy as np
import pandas as pd

def assert_clean_feature_space(feature_cols):
    """
    Ensures discrete 'key' and 'mode' binary/categorical fields are strictly 
    excluded from vector calculations to prevent distorted Euclidean distances.
    """
    forbidden_cols = {'key', 'mode', 'key_x', 'key_y'}
    found_forbidden = forbidden_cols.intersection(set(feature_cols))
    
    if found_forbidden:
        raise ValueError(
            f"❌ Vector space violation! Found forbidden fields {found_forbidden}. "
            "Discrete key/mode fields distort distance calculations and must be removed."
        )


def apply_scoring_rule(seed_vector, feature_matrix):
    """
    Calculates Euclidean distance between the seed vector and all candidates in RAM,
    then converts distance into a normalized similarity percentage score.
    """
    # 1. Compute Euclidean distance across 9D continuous feature space
    distances = np.linalg.norm(feature_matrix - seed_vector, axis=1)
    
    # 2. Scale distance realistically. 
    # In a 9D normalized space, realistic max distance between non-extreme tracks is ~1.8
    effective_max_dist = 1.8
    
    # 3. Compute normalized match score (0.0 to 100.0)
    scores = np.maximum(0.0, 100.0 * (1.0 - (distances / effective_max_dist)))
    return scores


def generate_feature_differentials(seed_vector, candidate_vector, feature_cols):
    """
    Calculates absolute distance per feature to identify the top drivers 
    of similarity between the seed track and a candidate recommendation.
    """
    assert_clean_feature_space(feature_cols)
    
    # Compute absolute difference per feature dimension
    deltas = np.abs(seed_vector - candidate_vector)
    
    # Map deltas to alignment scores (1.0 = identical match, 0.0 = total opposite)
    alignment = {col: float(1.0 - delta) for col, delta in zip(feature_cols, deltas)}
    
    # Sort features by highest alignment score
    sorted_drivers = sorted(alignment.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 3 drivers and raw alignment dictionary
    return sorted_drivers[:3], alignment


def apply_ranking_rule(scores, df, top_n=5):
    """
    Uses fast NumPy argpartition to select the top_n results instantly
    without copying or sorting the entire 1.2M row DataFrame in RAM.
    """
    # 1. Retrieve top_n candidate indices using fast partial sorting
    if len(scores) > top_n:
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        # Sort only the top_n subset in descending order
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    else:
        top_indices = np.argsort(-scores)

    # 2. Slice the dataframe for only those top indices
    ranked_df = df.iloc[top_indices].copy()
    ranked_df['match_score'] = np.round(scores[top_indices], 1)
    
    return ranked_df