"""
src/app/streamlit_app.py

Streamlit demo UI for the IT-support-ticket routing project.

This file is a thin presentation layer that ties together three already-working
layers of the project:

    1. Classification  -> CASCADE classifier (src/classification/train_cascade.py)
                           Tier-1 = TF-IDF + LogisticRegression (fast path,
                             trained at startup - this is fast even on 4000 rows)
                           Tier-2 = MiniLM embeddings + LogisticRegression,
                             loaded from models/ticket_classifier.joblib
                             (escalation path, used when Tier-1 is unsure)
    2. RAG retrieval    -> src/rag/suggest_resolution.py
                           (retrieve_similar_tickets)
    3. Grounded answer  -> src/rag/suggest_resolution.py
                           (build_llm_prompt + call_gemini), with escalation
                           when the best match is below SIMILARITY_THRESHOLD.

It does NOT reimplement any pipeline logic; it imports and reuses the existing
functions/constants from suggest_resolution.py and train_cascade.py.

Run from the project root with:
    streamlit run src/app/streamlit_app.py
"""

import os
import sys
import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
# Project-wide convention: project root is two directories up from this file.
# This file lives at src/app/streamlit_app.py, so:
#   this file -> src/app -> src -> <project root>
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))

# Make sure the project root is importable so `from src.rag...` works no matter
# what directory Streamlit is launched from.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Key file locations (single source of truth for the loader + error messages).
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "ticket_index.faiss")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "ticket_metadata.json")
CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "ticket_classifier.joblib")
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "synthetic_tickets.csv")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini"  # display label; actual model id lives in suggest_resolution.py

# Applied cascade confidence threshold.
#
# NOT re-derived at app startup (calibration is slow and would make the demo
# sluggish). This is the value derived by the calibration methodology in
# src/classification/train_cascade.py (175-ticket paraphrased calibration set,
# 70-80% Tier-1 target-accuracy band - see RESULTS.md for the full derivation
# and the accuracy/efficiency tradeoff analysis behind this number).
CASCADE_CONFIDENCE_THRESHOLD = 0.50

# The 3 canonical demo examples (kept in sync with suggest_resolution.py's demo):
#   1. VPN issue           -> should classify + retrieve + suggest
#   2. Password reset      -> should classify + retrieve + suggest
#   3. Weather question    -> deliberately unrelated -> should escalate
DEMO_EXAMPLES = [
    {
        "label": "🔌 VPN issue",
        "title": "Cannot connect to company VPN",
        "description": (
            "Since this morning I keep getting a timeout error when I try to "
            "connect to the corporate VPN from home. The client says "
            "'connection failed' after about 30 seconds. Restarting the client "
            "and my router did not help."
        ),
    },
    {
        "label": "🔑 Password reset",
        "title": "Need to reset my account password",
        "description": (
            "I am locked out of my account after too many failed login "
            "attempts and I do not remember my current password. Please help me "
            "reset it so I can log back in."
        ),
    },
    {
        "label": "🌦️ Unrelated (weather)",
        "title": "What is the weather like today?",
        "description": (
            "I just wanted to know whether it is going to rain this afternoon "
            "and if I should bring an umbrella to the office."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Page configuration  (MUST be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="IT Ticket Routing & Resolution Agent",
    page_icon="🎫",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached resource loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_resources():
    """
    Load every heavy/one-time resource ONCE and cache it across Streamlit reruns.

    Returns a dict on success. On any missing-prerequisite condition it returns
    a dict with an "error" key describing the problem and the exact remediation
    step, so the UI layer can render a clean st.error() + st.stop() instead of
    crashing with a raw traceback in front of an audience.
    """
    # Imports are done inside the cached function so that if a dependency (or a
    # file) is missing, we surface a friendly message rather than a hard import
    # crash at module load time.
    try:
        from dotenv import load_dotenv
        import faiss  # noqa: F401  (imported to fail fast if faiss is missing)
        import joblib
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - environment/setup issue
        return {
            "error": (
                "A required Python package failed to import "
                f"({type(exc).__name__}: {exc}).\n\n"
                "Install the project dependencies first, e.g.:\n"
                "    pip install -r requirements.txt"
            )
        }

    # --- .env / GEMINI_API_KEY ------------------------------------------- #
    # suggest_resolution.py reads GEMINI_API_KEY from .env via python-dotenv.
    # We load it here too so the key is available in the environment, and we
    # check for its presence up front to fail loud-but-friendly.
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()  # fall back to any .env on the default search path

    if not os.getenv("GEMINI_API_KEY"):
        return {
            "error": (
                "GEMINI_API_KEY is not set.\n\n"
                f"Create a `.env` file at the project root ({ENV_PATH}) "
                "containing:\n"
                "    GEMINI_API_KEY=your_api_key_here"
            )
        }

    # --- FAISS index ------------------------------------------------------ #
    if not os.path.exists(FAISS_INDEX_PATH):
        return {
            "error": (
                "FAISS vector index not found at:\n"
                f"    {FAISS_INDEX_PATH}\n\n"
                "Build it first by running:\n"
                "    python src/rag/build_vector_index.py"
            )
        }

    # --- Metadata JSON ---------------------------------------------------- #
    if not os.path.exists(METADATA_PATH):
        return {
            "error": (
                "Ticket metadata not found at:\n"
                f"    {METADATA_PATH}\n\n"
                "Build it first by running:\n"
                "    python src/rag/build_vector_index.py"
            )
        }

    # --- Tier-2 classifier (MiniLM + LogReg, saved artifact) --------------- #
    if not os.path.exists(CLASSIFIER_PATH):
        return {
            "error": (
                "Trained Tier-2 classifier not found at:\n"
                f"    {CLASSIFIER_PATH}\n\n"
                "Train it first by running:\n"
                "    python src/classification/train_embeddings.py"
            )
        }

    # --- Dataset CSV (needed to train Tier-1 at startup) ------------------- #
    if not os.path.exists(DATA_CSV):
        return {
            "error": (
                "Ticket dataset not found at:\n"
                f"    {DATA_CSV}\n\n"
                "This is needed to train the Tier-1 (TF-IDF) fast-path model "
                "at startup. Generate it first by running:\n"
                "    python data/generate_dataset.py"
            )
        }

    # --- Reused RAG layer ------------------------------------------------- #
    # Import the already-working functions/constants rather than reimplementing.
    try:
        from src.rag import suggest_resolution as sr
    except Exception:
        # Fall back to importing as a top-level module if the project isn't
        # laid out as a package (no __init__.py). Add src/rag to the path.
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "rag"))
            import suggest_resolution as sr  # type: ignore
        except Exception as exc:
            return {
                "error": (
                    "Could not import src/rag/suggest_resolution.py "
                    f"({type(exc).__name__}: {exc}).\n\n"
                    "Make sure the file exists and that you are running "
                    "Streamlit from the project root:\n"
                    "    streamlit run src/app/streamlit_app.py"
                )
            }

    # --- Reused cascade classifier layer ------------------------------------ #
    # train_tier1 / get_tier1_confidence come from train_cascade.py and are
    # reused as-is (same TF-IDF config, same confidence-extraction logic as
    # every calibration/evaluation run already documented in RESULTS.md).
    try:
        from src.classification.train_cascade import train_tier1, get_tier1_confidence
    except Exception as exc:
        return {
            "error": (
                "Could not import src/classification/train_cascade.py "
                f"({type(exc).__name__}: {exc}).\n\n"
                "Make sure the file exists and that you are running "
                "Streamlit from the project root:\n"
                "    streamlit run src/app/streamlit_app.py"
            )
        }

    # --- Load the heavy artifacts ---------------------------------------- #
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        tier2_classifier = joblib.load(CLASSIFIER_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as exc:
        return {
            "error": (
                "Failed while loading model/classifier/metadata/index "
                f"({type(exc).__name__}: {exc}).\n\n"
                "Verify the artifacts were produced by the current pipeline "
                "and are not corrupted, then rerun the build/train scripts."
            )
        }

    # --- Train Tier-1 (TF-IDF + LogReg) at startup ------------------------- #
    # No saved artifact exists for Tier-1 - it trains in seconds even on 4000
    # rows, so training here on every app start is fine (unlike Tier-2, which
    # is loaded from disk to avoid the ~100s MiniLM embedding step).
    try:
        df = pd.read_csv(DATA_CSV)
        required_cols = {"title", "description", "category"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return {
                "error": (
                    f"{os.path.basename(DATA_CSV)} is missing required "
                    f"column(s): {sorted(missing_cols)}"
                )
            }
        tier1_texts = (
            df["title"].fillna("").astype(str)
            + " "
            + df["description"].fillna("").astype(str)
        ).tolist()
        tier1_labels = df["category"].astype(str).tolist()
        tier1_vectorizer, tier1_classifier = train_tier1(tier1_texts, tier1_labels)
    except Exception as exc:
        return {
            "error": (
                "Failed to train the Tier-1 (TF-IDF) fast-path classifier "
                f"({type(exc).__name__}: {exc})."
            )
        }

    # --- Gemini client ------------------------------------------------------ #
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as exc:
        return {
            "error": (
                "Failed to initialize the Gemini client "
                f"({type(exc).__name__}: {exc}).\n\n"
                "Check that GEMINI_API_KEY in your .env is a valid key from "
                "https://aistudio.google.com/apikey"
            )
        }

    similarity_threshold = getattr(sr, "SIMILARITY_THRESHOLD", 0.35)

    return {
        "model": model,
        "tier1_vectorizer": tier1_vectorizer,
        "tier1_classifier": tier1_classifier,
        "tier2_classifier": tier2_classifier,
        "get_tier1_confidence": get_tier1_confidence,
        "metadata": metadata,
        "index": index,
        "faiss": faiss,
        "client": client,
        "sr": sr,
        "similarity_threshold": similarity_threshold,
    }


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def combined_text(title: str, description: str) -> str:
    """Project-wide input convention: title + ' ' + description."""
    return f"{title} {description}".strip()


def classify_ticket_cascade(
    text: str,
    tier1_vectorizer,
    tier1_classifier,
    tier2_classifier,
    embedding_model,
    get_tier1_confidence_fn,
):
    """
    Run the confidence-based cascade: Tier-1 (TF-IDF) resolves the ticket if
    confident enough; otherwise escalate to Tier-2 (MiniLM embeddings).

    Returns a dict: category, confidence (of whichever tier resolved it),
    tier (1 or 2), tier1_pred, tier1_conf (always reported, even on escalation,
    for the "why did this escalate" explanation in the UI).
    """
    tier1_preds, tier1_confs = get_tier1_confidence_fn(
        tier1_vectorizer, tier1_classifier, [text]
    )
    tier1_pred = str(np.asarray(tier1_preds).ravel()[0])
    tier1_conf = float(np.asarray(tier1_confs).ravel()[0])

    if tier1_conf >= CASCADE_CONFIDENCE_THRESHOLD:
        return {
            "category": tier1_pred,
            "confidence": tier1_conf,
            "tier": 1,
            "tier1_pred": tier1_pred,
            "tier1_conf": tier1_conf,
        }

    # Escalate to Tier-2.
    embedding = embedding_model.encode([text])
    embedding = np.asarray(embedding, dtype=np.float32)
    tier2_pred = tier2_classifier.predict(embedding)[0]

    tier2_conf = None
    if hasattr(tier2_classifier, "predict_proba"):
        proba = tier2_classifier.predict_proba(embedding)[0]
        classes = list(tier2_classifier.classes_)
        try:
            idx = classes.index(tier2_pred)
            tier2_conf = float(proba[idx])
        except ValueError:
            tier2_conf = float(np.max(proba))

    return {
        "category": tier2_pred,
        "confidence": tier2_conf,
        "tier": 2,
        "tier1_pred": tier1_pred,
        "tier1_conf": tier1_conf,
    }


def truncate(text, length: int = 120) -> str:
    """Truncate a resolution preview for the results table."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


def category_badge(category: str, confidence, tier: int) -> None:
    """Render the predicted category prominently, with confidence + which tier."""
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #1f77b4 0%, #2a9df4 100%);
                color: white;
                padding: 18px 22px;
                border-radius: 12px;
                font-size: 1.4rem;
                font-weight: 700;
                text-align: center;">
                🏷️ {category}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        if confidence is not None:
            st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
        else:
            st.metric(label="Confidence", value="n/a")
    with col_c:
        tier_label = "Tier-1 (fast)" if tier == 1 else "Tier-2 (escalated)"
        st.metric(label="Resolved by", value=tier_label)


# --------------------------------------------------------------------------- #
# Load resources (with spinner) or bail out cleanly
# --------------------------------------------------------------------------- #
with st.spinner("Loading models and index (first run only)…"):
    resources = load_resources()

if "error" in resources:
    st.title("🎫 IT Ticket Routing & Resolution Agent")
    st.error("**Cannot start the demo — a prerequisite is missing:**")
    st.code(resources["error"], language="text")
    st.info(
        "Fix the item above and reload this page. Heavy resources are cached, "
        "so a successful load only happens once per session."
    )
    st.stop()

MODEL = resources["model"]
TIER1_VECTORIZER = resources["tier1_vectorizer"]
TIER1_CLASSIFIER = resources["tier1_classifier"]
TIER2_CLASSIFIER = resources["tier2_classifier"]
GET_TIER1_CONFIDENCE = resources["get_tier1_confidence"]
METADATA = resources["metadata"]
INDEX = resources["index"]
FAISS = resources["faiss"]
CLIENT = resources["client"]
SR = resources["sr"]
SIMILARITY_THRESHOLD = resources["similarity_threshold"]


# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
def _init_state():
    defaults = {
        "title_input": "",
        "desc_input": "",
        "results": None,  # holds the last pipeline output dict
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _apply_example(example: dict):
    """Pre-fill the input widgets from a demo example and clear old results."""
    st.session_state["title_input"] = example["title"]
    st.session_state["desc_input"] = example["description"]
    st.session_state["results"] = None


# --------------------------------------------------------------------------- #
# Sidebar — System Info
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ System Info")
    n_tickets = len(METADATA) if isinstance(METADATA, (list, dict)) else "n/a"
    st.markdown(
        f"""
        - **Tickets indexed:** `{n_tickets}`
        - **Tier-1 (fast path):** `TF-IDF + LogisticRegression`
        - **Tier-2 (escalation):** `{EMBEDDING_MODEL_NAME}` embeddings + LogisticRegression
        - **LLM:** `Google Gemini`
        - **Cascade confidence threshold:** `{CASCADE_CONFIDENCE_THRESHOLD:.2f}`
        - **RAG similarity threshold:** `{SIMILARITY_THRESHOLD:.2f}`
        - **Categories:** 7
        """
    )
    st.caption(
        "Cascade threshold derived via calibration on a 175-ticket paraphrased "
        "set (see RESULTS.md). Below the RAG similarity threshold, the agent "
        "escalates to a human instead of calling the LLM."
    )


# --------------------------------------------------------------------------- #
# Header + intro
# --------------------------------------------------------------------------- #
st.title("🎫 IT Ticket Routing & Resolution Agent")
st.markdown(
    "Enter a new IT support ticket to see the full pipeline run: "
    "**cascade classification → similar-ticket retrieval → grounded resolution "
    "(or human escalation).**"
)

with st.expander("ℹ️ How this works"):
    st.markdown(
        """
        This demo runs a **three-layer** support-triage pipeline:

        1. **Cascade classification** — a fast TF-IDF model (Tier-1) handles
           the ticket if it's confident enough; otherwise the ticket is
           **escalated** to a stronger MiniLM-embeddings model (Tier-2). This
           mirrors the same "don't guess when unsure" philosophy used one
           layer down in Step 3.
        2. **RAG retrieval** — the ticket text is embedded and matched
           against a FAISS index of past tickets to surface the most
           similar resolved cases.
        3. **Agentic escalation** — if the best match is confident enough,
           Gemini drafts a resolution *grounded in those retrieved tickets*;
           if nothing is similar enough (below the threshold), the ticket is
           automatically **escalated to a human agent** instead.
        """
    )


# --------------------------------------------------------------------------- #
# Input section
# --------------------------------------------------------------------------- #
st.subheader("📝 New Ticket")

st.caption("Quick-fill a demo example:")
ex_cols = st.columns(len(DEMO_EXAMPLES))
for col, example in zip(ex_cols, DEMO_EXAMPLES):
    with col:
        st.button(
            example["label"],
            use_container_width=True,
            on_click=_apply_example,
            args=(example,),
            key=f"ex_{example['label']}",
        )

st.text_input("Ticket title", key="title_input", placeholder="Short summary of the issue")
st.text_area(
    "Ticket description",
    key="desc_input",
    height=140,
    placeholder="Describe the problem in detail…",
)

analyze = st.button("🚀 Analyze Ticket", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Pipeline execution (only on button click)
# --------------------------------------------------------------------------- #
def run_pipeline(title: str, description: str) -> dict:
    """
    Execute the full pipeline and return a serializable results dict that is
    stored in session_state so the display survives Streamlit reruns.

    Any Gemini-specific error is mapped to the same categories established in
    suggest_resolution.py and returned as a ('error_kind', message) pair so the
    UI can render actionable st.error() guidance instead of a stack trace.
    """
    text = combined_text(title, description)

    # (a) Cascade classification
    classification = classify_ticket_cascade(
        text,
        TIER1_VECTORIZER,
        TIER1_CLASSIFIER,
        TIER2_CLASSIFIER,
        MODEL,
        GET_TIER1_CONFIDENCE,
    )

    # (b) Retrieval — reuse the existing function; be defensive about sorting.
    retrieved = SR.retrieve_similar_tickets(text, MODEL, INDEX, METADATA, FAISS, top_k=5) or []
    retrieved = sorted(
        retrieved, key=lambda r: r.get("similarity", 0.0), reverse=True
    )

    top_similarity = retrieved[0]["similarity"] if retrieved else 0.0

    result = {
        "category": classification["category"],
        "confidence": classification["confidence"],
        "tier": classification["tier"],
        "tier1_pred": classification["tier1_pred"],
        "tier1_conf": classification["tier1_conf"],
        "retrieved": retrieved,
        "top_similarity": top_similarity,
        "escalated": False,
        "suggestion": None,
        "llm_error": None,  # (kind, message)
    }

    # (c) Suggestion vs. escalation — same threshold logic as suggest_resolution.
    if top_similarity < SIMILARITY_THRESHOLD:
        result["escalated"] = True
        return result

    # Build the grounded prompt and call Gemini, reusing the existing helpers.
    try:
        prompt = SR.build_llm_prompt(title, description, retrieved)
        suggestion = SR.call_gemini(CLIENT, prompt)

        if not suggestion or not str(suggestion).strip():
            result["llm_error"] = (
                "empty",
                "The LLM returned an empty or blocked response. This can happen "
                "if the content was filtered. Try rephrasing the ticket, or "
                "escalate to a human agent.",
            )
        else:
            result["suggestion"] = str(suggestion).strip()

    except Exception as exc:  # map to the categories used in suggest_resolution
        msg = f"{exc}".lower()
        if "api" in msg and "key" in msg or "credential" in msg:
            kind, guidance = (
                "auth",
                "Authentication failed. Check that GEMINI_API_KEY in your .env "
                "file is valid and has not expired.",
            )
        elif "rate" in msg or "quota" in msg or "429" in msg:
            kind, guidance = (
                "rate_limit",
                "Gemini rate limit / quota exceeded. Wait a moment and try "
                "again, or check your API usage quota.",
            )
        elif "network" in msg or "connection" in msg or "timeout" in msg or "dns" in msg:
            kind, guidance = (
                "network",
                "Network error reaching the Gemini API. Check your internet "
                "connection and try again.",
            )
        else:
            kind, guidance = (
                "unknown",
                f"Unexpected error while generating the suggestion: {exc}",
            )
        result["llm_error"] = (kind, guidance)

    return result


if analyze:
    title = st.session_state["title_input"].strip()
    description = st.session_state["desc_input"].strip()

    if not title and not description:
        st.warning("Please enter a ticket title and/or description first.")
    else:
        # Whole-pipeline safety net: never let a raw traceback hit the panel.
        try:
            with st.spinner("Running the pipeline…"):
                st.session_state["results"] = run_pipeline(title, description)
        except Exception as exc:
            st.session_state["results"] = None
            st.error(
                "❌ The pipeline hit an unexpected error and stopped safely.\n\n"
                f"**{type(exc).__name__}:** {exc}"
            )
            with st.expander("Technical details (for debugging)"):
                st.code(traceback.format_exc(), language="text")


# --------------------------------------------------------------------------- #
# Results display  (reads from session_state so it persists across reruns)
# --------------------------------------------------------------------------- #
results = st.session_state.get("results")

if results is not None:
    st.divider()

    # ---- (a) Classification ------------------------------------------- #
    st.subheader("1️⃣ Classification (Cascade)")
    category_badge(results["category"], results["confidence"], results["tier"])

    if results["tier"] == 2:
        st.info(
            f"Tier-1's confidence (**{results['tier1_conf']:.2f}**) was below the "
            f"calibrated threshold (**{CASCADE_CONFIDENCE_THRESHOLD:.2f}**), so "
            "this ticket was **escalated** to the more accurate MiniLM "
            f"embeddings model. Tier-1 would have guessed "
            f"**{results['tier1_pred']}**; Tier-2 predicted "
            f"**{results['category']}**."
        )
    else:
        st.success(
            f"Tier-1's confidence (**{results['tier1_conf']:.2f}**) met the "
            f"calibrated threshold (**{CASCADE_CONFIDENCE_THRESHOLD:.2f}**), so "
            "the cheap fast-path model handled this ticket — no escalation needed."
        )

    # ---- (b) Retrieval ------------------------------------------------ #
    st.subheader("2️⃣ Similar Past Tickets (RAG retrieval)")
    retrieved = results["retrieved"]
    if retrieved:
        rows = []
        for r in retrieved:
            sim = r.get("similarity", 0.0)
            rows.append(
                {
                    "Similarity": f"{sim * 100:.1f}%",
                    "Category": r.get("category", "—"),
                    "Title": r.get("title", "—"),
                    "Resolution (preview)": truncate(r.get("resolution", "")),
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            f"Top match similarity: **{results['top_similarity'] * 100:.1f}%** "
            f"(threshold: {SIMILARITY_THRESHOLD * 100:.0f}%)"
        )
    else:
        st.info("No similar tickets were returned from the index.")

    # ---- (c) Suggestion or escalation --------------------------------- #
    st.subheader("3️⃣ Resolution")

    if results["escalated"]:
        st.warning(
            "⚠️ **Escalated to a human agent.** No sufficiently similar past "
            "ticket was found (top similarity "
            f"{results['top_similarity'] * 100:.1f}% is below the "
            f"{SIMILARITY_THRESHOLD * 100:.0f}% threshold), so the agent did "
            "not attempt an automated resolution."
        )
    elif results["llm_error"] is not None:
        _, guidance = results["llm_error"]
        st.error(f"❌ Could not generate a suggestion.\n\n{guidance}")
    elif results["suggestion"]:
        st.info("🤖 **Suggested resolution (grounded in retrieved tickets):**")
        st.markdown(results["suggestion"])
    else:
        st.info("No suggestion was produced.")