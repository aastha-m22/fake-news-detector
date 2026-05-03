"""
app.py — Fake News Detection System: Streamlit web application.

Design aesthetic: editorial newspaper meets dark forensic terminal.
Clean, purposeful, investigative. Every element earns its place.

Run with: streamlit run app.py
"""

import os
import time
import streamlit as st
from model import get_or_train_model, predict
from explain import explain_prediction, highlight_text, build_highlighted_html

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS — dark editorial / forensic newsroom aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:wght@300;400;600&display=swap');

/* ── Reset & base ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: #0a0a0a;
    color: #e8e0d0;
}
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; }

/* ── Masthead ──────────────────────────────────── */
.masthead {
    border-top: 5px solid #e8e0d0;
    border-bottom: 3px solid #e8e0d0;
    padding: 1.5rem 0 1.2rem;
    margin-bottom: 0.5rem;
    text-align: center;
    position: relative;
}
.masthead-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #8a7f70;
    margin-bottom: 0.5rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 900;
    letter-spacing: -1px;
    color: #f0e8d8;
    line-height: 1;
    margin: 0;
}
.masthead-rule {
    border: none;
    border-top: 1px solid #333;
    margin: 1rem 0 0;
}
.masthead-tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5a5040;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ── Section labels ────────────────────────────── */
.section-rule {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.8rem 0 1.2rem;
}
.section-rule-line { flex: 1; height: 1px; background: #2a2a2a; }
.section-rule-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #5a5040;
    white-space: nowrap;
}

/* ── Verdict card ──────────────────────────────── */
.verdict-card {
    border: 2px solid;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.verdict-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.verdict-card.fake {
    border-color: #7f1d1d;
    background: linear-gradient(135deg, #1c0505, #140000);
}
.verdict-card.fake::before { background: #ef4444; }
.verdict-card.real {
    border-color: #14532d;
    background: linear-gradient(135deg, #021505, #010d02);
}
.verdict-card.real::before { background: #22c55e; }

.verdict-stamp {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 4px;
    text-transform: uppercase;
    line-height: 1;
}
.verdict-stamp.fake { color: #ef4444; text-shadow: 0 0 40px rgba(239,68,68,0.3); }
.verdict-stamp.real { color: #22c55e; text-shadow: 0 0 40px rgba(34,197,94,0.3); }

.verdict-confidence {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8a7f70;
    letter-spacing: 1.5px;
    margin-top: 0.25rem;
}
.verdict-bar-track {
    background: #1a1a1a;
    border-radius: 2px;
    height: 4px;
    margin-top: 1rem;
    overflow: hidden;
}
.verdict-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s ease;
}

/* ── Highlighted article ───────────────────────── */
.article-highlight-box {
    background: #0f0f0f;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    font-family: 'Source Serif 4', serif;
    font-size: 1.0rem;
    line-height: 1.85;
    color: #c8c0b0;
    max-height: 300px;
    overflow-y: auto;
    position: relative;
}
.highlight-legend {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
}
.legend-item { display: flex; align-items: center; gap: 0.4rem; }
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 2px;
    display: inline-block;
}

/* ── LIME word bars ────────────────────────────── */
.lime-word-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #141414;
}
.lime-word {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #d0c8b8;
    min-width: 130px;
    max-width: 130px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.lime-bar-track { flex: 1; background: #111; border-radius: 2px; height: 6px; }
.lime-bar-fill { height: 100%; border-radius: 2px; }
.lime-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5a5040;
    min-width: 50px;
    text-align: right;
}

/* ── Metric cells ──────────────────────────────── */
.metric-cell {
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
    background: #0d0d0d;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8e0d0;
    line-height: 1;
}
.metric-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4a4030;
    margin-top: 0.3rem;
}

/* ── Textarea ──────────────────────────────────── */
.stTextArea textarea {
    background: #080808 !important;
    border: 1px solid #2a2520 !important;
    border-radius: 4px !important;
    color: #d0c8b0 !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    caret-color: #8a7f70;
}
.stTextArea textarea:focus {
    border-color: #4a4030 !important;
    box-shadow: 0 0 0 1px #3a3020 !important;
}

/* ── Analyze button ─────────────────────────────── */
.stButton > button {
    background: #e8e0d0 !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2.5rem !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #f8f0e0 !important;
    transform: translateY(-1px) !important;
}

/* ── Info / warning ─────────────────────────────── */
.dispatch-box {
    border-left: 3px solid #3a3020;
    padding: 0.7rem 1rem;
    background: #0c0c0a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #7a7060;
    line-height: 1.6;
    margin: 0.6rem 0;
}

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #2a2520; border-radius: 2px; }

/* ── Expander ────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1a1a1a !important;
    background: #0a0a0a !important;
    border-radius: 4px !important;
}
details > summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #6a6050 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "result" not in st.session_state:
    st.session_state.result = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# ---------------------------------------------------------------------------
# Lazy model loader (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached():
    return get_or_train_model(data_source="sample")


# ---------------------------------------------------------------------------
# Helper: render section rule
# ---------------------------------------------------------------------------
def section_rule(label: str):
    st.markdown(
        f"""<div class="section-rule">
              <div class="section-rule-line"></div>
              <div class="section-rule-label">{label}</div>
              <div class="section-rule-line"></div>
            </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helper: render LIME word bars
# ---------------------------------------------------------------------------
def render_word_bars(words: list, color: str, max_score: float = 1.0):
    if not words:
        st.markdown(
            '<div class="dispatch-box">No significant words detected.</div>',
            unsafe_allow_html=True,
        )
        return

    max_s = max((s for _, s in words), default=1.0) or 1.0
    rows_html = ""
    for word, score in words[:8]:
        pct = (score / max_s) * 100
        rows_html += f"""
        <div class="lime-word-row">
          <div class="lime-word">{word}</div>
          <div class="lime-bar-track">
            <div class="lime-bar-fill"
                 style="width:{pct:.1f}%;background:{color}"></div>
          </div>
          <div class="lime-score">{score:.3f}</div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sample articles for quick testing
# ---------------------------------------------------------------------------
SAMPLE_FAKE_TEXT = (
    "BREAKING: Scientists have confirmed that 5G towers emit a special frequency "
    "that activates the tracking nanobots secretly inserted through COVID vaccines. "
    "A whistleblower at a major pharmaceutical company leaked internal documents "
    "showing global elites plan to use this technology to control human behavior "
    "by 2025. Mainstream media is burying this explosive story."
)

SAMPLE_REAL_TEXT = (
    "NASA's James Webb Space Telescope has captured a stunning image of the Carina "
    "Nebula, revealing previously invisible star-forming regions. The image, released "
    "by the agency on Tuesday, shows hundreds of new stars never seen before and "
    "represents the sharpest infrared image of this region ever taken, according to "
    "officials at the Space Telescope Science Institute in Baltimore."
)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ── Masthead ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="masthead">
        <div class="masthead-eyebrow">Est. 2024 · AI-Powered Journalism Tool</div>
        <div class="masthead-title">The Fake News Detector</div>
        <hr class="masthead-rule">
        <div class="masthead-tagline">NLP · Passive Aggressive Classifier · LIME Explainability</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────────────────────
    with st.spinner("Initializing the analysis engine..."):
        try:
            model = load_model_cached()
            st.session_state.model = model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    # ── Two-column layout ─────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        section_rule("SUBMIT ARTICLE FOR ANALYSIS")

        news_text = st.text_area(
            "Article text",
            value=st.session_state.input_text,
            height=220,
            placeholder="Paste the full news article or headline here...",
            label_visibility="collapsed",
        )

        # Quick-load sample buttons
        q1, q2 = st.columns(2)
        with q1:
            if st.button("📰 Load Sample — REAL", use_container_width=True):
                st.session_state.input_text = SAMPLE_REAL_TEXT
                st.rerun()
        with q2:
            if st.button("⚠️ Load Sample — FAKE", use_container_width=True):
                st.session_state.input_text = SAMPLE_FAKE_TEXT
                st.rerun()

        # Sync text area value after button clicks
        if st.session_state.input_text and not news_text:
            news_text = st.session_state.input_text

        st.write("")
        analyze_btn = st.button("🔍 ANALYZE ARTICLE", use_container_width=True)

    with right_col:
        section_rule("INTELLIGENCE BRIEF")
        st.markdown("""
        <div class="dispatch-box">
            This system combines TF-IDF vectorization with a Passive Aggressive
            Classifier to detect fake news patterns, then applies LIME to reveal
            which specific words drove the verdict.
        </div>
        <div class="dispatch-box">
            <strong>Red highlights</strong> → words pushing toward FAKE<br>
            <strong>Green highlights</strong> → words pushing toward REAL<br>
            Confidence is derived from the model's decision boundary distance.
        </div>
        """, unsafe_allow_html=True)

        # Tiny model stats
        section_rule("MODEL STATUS")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="metric-cell">
                <div class="metric-val">PAC</div>
                <div class="metric-lbl">Classifier</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="metric-cell">
                <div class="metric-val">TF-IDF</div>
                <div class="metric-lbl">Vectorizer</div>
            </div>""", unsafe_allow_html=True)

    # ── Run analysis ──────────────────────────────────────────────────────────
    if analyze_btn:
        if not news_text or len(news_text.strip()) < 20:
            st.warning("Please enter at least 20 characters of news text.")
        else:
            st.session_state.input_text = news_text

            with st.spinner("Running classifier and generating LIME explanation..."):
                t0 = time.time()
                result = predict(model, news_text)
                explanation = explain_prediction(
                    model, news_text, num_features=12, num_samples=400
                )
                elapsed = time.time() - t0

            st.session_state.result = result
            st.session_state.explanation = explanation
            st.session_state.elapsed = elapsed

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.result:
        result = st.session_state.result
        explanation = st.session_state.explanation
        elapsed = getattr(st.session_state, "elapsed", 0)

        section_rule("VERDICT")

        label = result["label"]
        conf = result["confidence"]
        css_class = label.lower()

        conf_bar_color = "#ef4444" if label == "FAKE" else "#22c55e"
        desc = (
            "High probability of misinformation detected. Exercise extreme caution."
            if label == "FAKE"
            else "Article characteristics align with credible reporting standards."
        )

        st.markdown(f"""
        <div class="verdict-card {css_class}">
            <div class="verdict-stamp {css_class}">{label}</div>
            <div class="verdict-confidence" style="margin-top:0.3rem">
                CONFIDENCE: {conf:.1f}% &nbsp;·&nbsp;
                REAL: {result['probabilities']['REAL']:.1f}% &nbsp;·&nbsp;
                FAKE: {result['probabilities']['FAKE']:.1f}%
            </div>
            <div style="font-family:'Source Serif 4',serif;font-size:0.88rem;
                        color:#8a7f70;margin-top:0.6rem;font-style:italic">
                {desc}
            </div>
            <div class="verdict-bar-track">
                <div class="verdict-bar-fill"
                     style="width:{min(conf,100):.1f}%;background:{conf_bar_color}"></div>
            </div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                    color:#3a3020;text-align:right;margin-top:0.3rem">
            Analysis completed in {elapsed:.2f}s
        </div>
        """, unsafe_allow_html=True)

        # ── LIME highlight section ────────────────────────────────────────────
        section_rule("LINGUISTIC FORENSICS — LIME ANALYSIS")

        ann_tokens = highlight_text(news_text, explanation)
        highlighted_html = build_highlighted_html(ann_tokens)

        st.markdown("**Highlighted Article** — hover over colored words to see influence scores")
        st.markdown(
            f'<div class="article-highlight-box">{highlighted_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div class="highlight-legend">
            <div class="legend-item">
                <div class="legend-dot" style="background:#ef4444"></div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                             color:#8a7f70;letter-spacing:1px">PROMOTES FAKE</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background:#22c55e"></div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                             color:#8a7f70;letter-spacing:1px">PROMOTES REAL</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # ── Word influence tables ─────────────────────────────────────────────
        lime_left, lime_right = st.columns(2, gap="large")

        with lime_left:
            st.markdown(
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
                'letter-spacing:2px;text-transform:uppercase;color:#7f1d1d;'
                'margin-bottom:0.5rem">⬆ Top FAKE Indicators</div>',
                unsafe_allow_html=True,
            )
            render_word_bars(explanation["words_positive"], "#ef4444")

        with lime_right:
            st.markdown(
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
                'letter-spacing:2px;text-transform:uppercase;color:#14532d;'
                'margin-bottom:0.5rem">⬆ Top REAL Indicators</div>',
                unsafe_allow_html=True,
            )
            render_word_bars(explanation["words_negative"], "#22c55e")

        # ── Raw LIME explanation (optional) ───────────────────────────────────
        with st.expander("RAW LIME EXPLANATION DATA"):
            if explanation.get("all_features"):
                import pandas as pd
                df = pd.DataFrame(
                    explanation["all_features"],
                    columns=["Word / Phrase", "LIME Weight (FAKE direction)"]
                )
                df = df.sort_values("LIME Weight (FAKE direction)", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div class="dispatch-box" style="margin-top:2rem">
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes.
            It is trained on a limited sample dataset and should not be used as the sole basis
            for determining the authenticity of any news article. Always consult primary sources
            and established fact-checking organizations.
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="border-top:1px solid #1a1a1a;margin-top:3rem;padding-top:1.2rem;
                text-align:center;font-family:'IBM Plex Mono',monospace;
                font-size:0.65rem;color:#2a2520;letter-spacing:2px;
                text-transform:uppercase">
        Fake News Detector · NLP + ML + LIME Explainability · Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
