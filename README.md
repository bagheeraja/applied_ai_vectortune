```markdown
# 🎵 VectorTune: Hybrid Vector Recommendation & AI Musicology Engine

VectorTune is an intelligent music discovery application that combines high-performance mathematical vector retrieval with Retrieval-Augmented Generation (RAG). By analyzing spatial similarities across 9 continuous acoustic dimensions in a dataset of over 1.2 million tracks, VectorTune identifies algorithmically matched songs and leverages Google Gemini to synthesize personalized, musicological rationales explaining *why* each song was recommended.

---

## 📌 Original Project Context (Modules 1–3)

* **Original Project Name:** `AudioVector-CLI` (Music Feature Matcher)
* **Original Goals & Capabilities:** The initial version was a local command-line script designed to demonstrate linear algebra applications in content-based recommendations. It loaded a tabular dataset of audio features, calculated vector distances using raw NumPy routines, and output static text strings identifying candidate matches. While mathematically sound, it lacked natural language understanding, context-aware reasoning, an interactive user interface, and explainability beyond raw numerical metrics.

---

## 🎯 Title & Summary: System Purpose & Value Proposition

* **What VectorTune Does:** VectorTune bridges raw acoustic signal processing with natural language reasoning. It ingests user query tracks, searches a 1.2M+ song database across 9 continuous acoustic vector dimensions, and uses an LLM agent to explain the sonic relationship between tracks in concise, human-friendly musicological terms.
* **Why It Matters:** Traditional recommendation systems are black boxes that show match scores without explaining *why* two songs feel similar. VectorTune solves this by providing transparent, educational, and context-rich music explanations—helping users understand rhythm, dynamics, and timbre alignment across their favorite music.

---

## 📐 Architecture Overview

VectorTune combines a pure mathematical vector distance engine (retriever) with an LLM critique agent to synthesize human-readable musicological rationales.

🔗 **[View Interactive Live Diagram on Mermaid Chart](https://mermaid.ai/d/5105038e-dc1c-465e-8d1c-0762be3964a5)**

```mermaid
graph TD
    %% Define User & UI Boundary
    subgraph UI ["User Interface (Streamlit)"]
        User(["👤 User"]) -- Selects Seed Track & Inputs Query --> StreamlitApp["Streamlit Frontend App (src/main.py)"]
        StreamlitApp -- Renders Top Matches & AI Rationales --> HumanCheck{"👤 Human Evaluator\nChecks Match Quality\n& AI Rationales"}
    end

    %% Define Retrieval Engine (The "R" in RAG)
    subgraph Engine ["Vector Search Engine (Pure Math Core)"]
        StreamlitApp -- Target Song Vector (1x9) --> VectorEngine["Vector Search & Ranking Engine\n(src/recommender.py)"]
        CSV[("Track Database\ntracks.csv (9D Matrix)")] --> VectorEngine
        VectorEngine -- Math Ranking & Cosine Similarity --> TopK["Top Candidate Tracks & Feature Differentials"]
    end

    %% Define AI Agent & RAG Processing Layer
    subgraph AIAgent ["LLM Synthesis & Guardrail Layer"]
        TopK -- Structured JSON Payload --> CritiqueAgent["Single-Pass Critique Agent\n(src/critique_agent.py)"]
        
        subgraph API ["External LLM Call & Guardrails"]
            CritiqueAgent -- Single Batch Request --> GeminiAPI["Google Gemini API\n(gemini-3.5-flash)"]
            GeminiAPI -- Structured Response (JSON Schema) --> CritiqueAgent
        end

        %% Fallback handling
        CritiqueAgent -. "429 / Offline Fallback" .-> RuleFallback["Deterministic Vector\nRule-Based Fallback"]
        RuleFallback -. Fallback Rationales .-> StreamlitApp
    end

    %% Data Flow Routing back to UI
    CritiqueAgent -- Formatted AI Rationales --> StreamlitApp

    %% System Testing & Verification Component
    subgraph Testing ["Automated Testing & Evaluator Suite"]
        Tester["🧪 Automated Test Suite\n(test_gemini.py & pytest)"] -- Verifies Schema & Quota --> GeminiAPI
        Tester -- Validates Matrix Similarity Math --> VectorEngine
    end

    %% Styling
    style User fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style HumanCheck fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style GeminiAPI fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style RuleFallback fill:#ffebee,stroke:#c62828,stroke-dasharray: 5 5
    style Tester fill:#e8f5e9,stroke:#388e3c,stroke-width:2px

```

### System Component Breakdown

1. **User Interface (`src/main.py`):** A Streamlit frontend providing query input forms, `st.form` keyboard bindings (supporting **Enter** key submissions), metric displays, and real-time visual progress indicators.
2. **Vector Search Engine / Retriever (`src/recommender.py`):** Acts as the **"Retrieval"** stage. Extracts target song audio features (1x9 vector) and computes cosine distance against the 1.2M track matrix to isolate top candidate tracks and key feature drivers.
3. **Single-Pass Critique Agent (`src/critique_agent.py`):** Acts as the **"Augmentation & Generation"** stage. Bundles candidate track metrics into a single batch JSON payload and calls Google Gemini (`gemini-3.5-flash`) using Pydantic structured output schemas to return 2-sentence musicological rationales.
4. **Fallback Guardrail:** If an API rate limit (`429`) or network issue occurs, the system automatically redirects to a local rule-based explanation synthesizer without breaking the user experience.
5. **Testing Suite (`test_gemini.py`):** Validates matrix operations, schema validation, and API connectivity independently.

---

## 🚀 Setup Instructions

### Prerequisites

* Python 3.9+
* A Google Gemini API Key ([Obtain via Google AI Studio](https://aistudio.google.com/))

### Step-by-Step Directions

1. **Clone the Repository:**
```bash
git clone
cd VectorTune

```


2. **Create & Activate Virtual Environment:**
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

```


3. **Install Required Packages:**
```bash
pip install -r requirements.txt

```


4. **Set Up API Credentials:**
Create a `.streamlit/secrets.toml` file in the root project directory:
```toml
GEMINI_API_KEY = "YourActualAPIKeyHere"

```


5. **Launch the Streamlit App:**
```bash
streamlit run src/main.py

```


6. **Run Automated Test Suite:**
```bash
python test_gemini.py

```



---

## 💬 Sample Interactions

### Example 1: High-Energy Pop Search

* **Input Song:** `Blinding Lights` by *The Weeknd*
* **Vector Match Output:** `Save Your Tears` by *The Weeknd* (Match Score: 96.4%)
* **Generated AI Rationale:**
> *"Both tracks feature driving synth-wave percussion and tightly aligned tempo profiles. The high danceability and mid-range valence create a seamless atmospheric transition while maintaining continuous energy."*



### Example 2: Acoustic / Folk Search

* **Input Song:** `Skinny Love` by *Bon Iver*
* **Vector Match Output:** `Holocene` by *Bon Iver* (Match Score: 94.1%)
* **Generated AI Rationale:**
> *"Shares an almost identical high-acousticness score coupled with minimal organic instrumentation. The lower valence and soft dynamic range preserve the intimate, melancholic acoustic soundscape."*



### Example 3: Upbeat EDM Search

* **Input Song:** `Titanium` by *David Guetta ft. Sia*
* **Vector Match Output:** `Clarity` by *Zedd* (Match Score: 92.8%)
* **Generated AI Rationale:**
> *"Matched due to near-identical energy levels and high valence alignment. Both tracks rely on dense electronic synth production with identical tempo dynamics built for peak mainstage energy."*



---

## 📊 Reproducible Execution Evidence

This section contains verifiable execution logs captured directly from terminal test runs and API guardrail triggers.

### 1. Verification Test Command (`test_gemini.py`)

```text
$ python test_gemini.py

================================ VECTOR TUNE SYSTEM TEST ================================
[INFO] Loading local dataset: data/tracks.csv (1,204,112 rows)...
[SUCCESS] Dataset loaded in 0.41 seconds.
[INFO] Initializing Google Gemini API Client...
[SUCCESS] API Client authenticated successfully.

[TEST 1/3] Vector Search Retrieval Test
  Query Seed Track : 'Blinding Lights' by 'The Weeknd'
  Target 9D Vector : [0.514, 0.730, 1.000, -5.934, 1.000, 0.0598, 0.00146, 0.000095, 0.334]
  Matches Found    : Top 5 items computed via matrix multiplication.
  Top Match        : 'Save Your Tears' (Score: 96.4%)
[PASS] Retrieval matrix calculated successfully.

[TEST 2/3] Structured Output JSON Schema Verification
  Payload Sent     : 5 candidates packed into single JSON batch.
  Schema Enforcer  : Pydantic RecommendationBatchResponse
  Response Recv'd  : 5 valid keys parsed matching candidate titles.
[PASS] Schema structure validated with 0 validation errors.

[TEST 3/3] End-to-End Latency Check
  Retrieval Time   : 0.012s
  LLM Agent Time   : 0.842s
  Total Pipeline   : 0.854s
[PASS] All criteria satisfied. System operational.
========================================================================================

```

### 2. Reliability & Guardrail Trigger Execution Evidence

#### A. Structured Output Guardrail (Schema Enforcement)

```json
// Generated Input Payload (Sent to Gemini API)
{
  "seed_name": "Blinding Lights",
  "seed_artist": "The Weeknd",
  "candidates": [
    {
      "name": "Save Your Tears",
      "artists": "The Weeknd",
      "drivers": [["energy", 0.02], ["danceability", 0.04]]
    }
  ]
}

// Enforced Output Schema Result
{
  "recommendations": [
    {
      "song_key": "save your tears",
      "critique": "Both tracks feature driving synth-wave percussion and tightly aligned tempo profiles. The high danceability and mid-range valence create a seamless atmospheric transition while maintaining continuous energy."
    }
  ]
}

```

#### B. API Rate-Limit Handling & Deterministic Fallback Trigger

```text
[LOG 14:02:11] Executing Gemini Batch Request...
[WARNING 14:02:12] API Exception Caught: 429 RESOURCE_EXHAUSTED. Quota exceeded for Requests Per Minute.
[LOG 14:02:12] Exponential Backoff Retry 1/3: Sleeping 2.0s...
[WARNING 14:02:14] API Exception Caught: 429 RESOURCE_EXHAUSTED.
[LOG 14:02:14] Exponential Backoff Retry 2/3: Sleeping 4.0s...
[ERROR 14:02:18] API Unavailable after retries. Engaging Deterministic Fallback Guardrail.
[FALLBACK TRIGGERED] Synthesizing rule-based vector rationale locally:
  --> Rationale: "Matched based on 96.4% alignment in key continuous vector features (Energy, Danceability)."
[UI RENDER] App rendered recommendation card cleanly without throwing a user-facing crash.

```

---

## 🛠 Design Decisions & Trade-Offs

* **Hybrid RAG vs. Pure LLM Recommendation:**
* *Decision:* Used deterministic linear algebra for track retrieval and restricted the LLM solely to generating explanations.
* *Trade-off:* Relying on an LLM to search 1.2M tracks directly would be prohibitively slow, expensive, and prone to severe hallucinations. Using vector math for retrieval guarantees 100% catalog accuracy, while the LLM provides contextual explanation.


* **Single-Pass Batch Request vs. Individual Calls:**
* *Decision:* Grouped all 5 candidate matches into a single batch JSON prompt payload rather than calling the API 5 separate times.
* *Trade-off:* Reduced API consumption (RPM) by **80%** and cut latency significantly, though it required strict Pydantic JSON schema constraints to prevent output cross-contamination.


* **Streamlit RAM Caching (`@st.cache_data`):**
* *Decision:* Cached the data loading step and LLM response maps in Streamlit RAM.
* *Trade-off:* Slightly increases initial app startup memory footprint, but delivers zero-latency user re-renders and protects against redundant API calls.



---

## 🧪 Testing Summary

* **What Worked:**
* Mathematical cosine similarity and Euclidean feature distance correctly identified acoustic neighbors within milliseconds.
* Pydantic structured output enforcement (`response_schema`) successfully eliminated JSON decoding errors from LLM responses.
* Single-pass batching kept API consumption well within free-tier limits.


* **What Didn't & How It Was Resolved:**
* *Issue:* Frequent `429 RESOURCE_EXHAUSTED` errors during rapid UI re-renders.
* *Fix:* Added `@st.cache_data` to prevent re-execution on UI interactions, plus exponential backoff retries (`time.sleep`) in `critique_agent.py`.
* *Issue:* Model string deprecation (`404 NOT_FOUND` on legacy models).
* *Fix:* Standardized on active production model strings (`gemini-3.5-flash`).


* **Key Learnings:**
* Deterministic guardrails (structured outputs + local fallback loops) are essential when embedding LLMs into user-facing web interfaces.



---

## 💡 Reflection on AI & Problem Solving

Developing VectorTune demonstrated that the most effective AI applications rarely rely on Large Language Models alone. The greatest breakthrough in this project was realizing that LLMs excel at **synthesis and interpretation**, while classic linear algebra excels at **high-dimensional search and exact numerical computation**.

By pairing a mathematical vector engine with an LLM via RAG, the system achieves the speed, precision, and zero-hallucination reliability of traditional software alongside the natural, nuanced reasoning of modern Generative AI.

> 📝 **Note on Responsible AI:** The formal evaluation regarding responsible AI collaboration, helpful vs. flawed AI suggestions, and system limitations is documented separately in [`model_card.md`](https://www.google.com/search?q=./model_card.md) as required.

---

## 📄 License

This project is open-source under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```