import streamlit as st

from src.predict import predict_news


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .result-card {
            padding: 1.2rem;
            border-radius: 12px;
            margin-top: 1rem;
        }
        .result-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .result-score {
            font-size: 1rem;
        }
        .footer {
            text-align: center;
            color: #8a8f98;
            font-size: 0.85rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown('<div class="main-title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">text classification </div>',
    unsafe_allow_html=True,
)

st.divider()




# ---------- Main input ----------
st.subheader("Analyze News Text")

user_text = st.text_area(
    "Paste the news article or news content below",
    height=260,
    placeholder=(
        "Enter text here"
    ),
    label_visibility="visible",
)

word_count = len(user_text.split()) if user_text.strip() else 0
st.caption(f"Word count: {word_count}")

if st.button("🔍 Analyze News", type="primary", use_container_width=True):
    text = user_text.strip()

    if not text:
        st.warning("Please enter some news text before analyzing.")
    elif word_count < 8:
        st.warning(
            "Please provide at least 8 words so the model has enough context for a prediction."
        )
    else:
        with st.spinner("Analyzing the news text..."):
            result, score = predict_news(text)

        st.divider()

        if result == 1:
            st.error("### ❌ FAKE NEWS")
            st.metric("Model score", f"{score:.2%}")
            st.caption(
                "The model classified this text as fake ."
            )
        elif result == 0:
            st.success("### ✅ REAL NEWS")
            st.metric("Model score", f"{score:.2%}")
            st.caption(
                "The model classified this text as real ."
            )
        else:
            st.warning("Unable to make a reliable prediction from the supplied text.")

st.divider()
st.markdown(
    '<div class="footer">Machine Learning Project • Fake News Detection</div>',
    unsafe_allow_html=True,
)
