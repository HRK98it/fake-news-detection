import streamlit as st
import tempfile

from src.predict import predict_news
from src.ocr import extract_text_from_image
from src.video_transcript import extract_text_from_video

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center;'>📰 Fake News Detection System</h1>
    <p style='text-align: center; color: gray;'>
    Detect fake news from Text, Images, and Videos using AI
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Image", "🎥 Video"])

# =====================================================
# 📝 TEXT TAB
# =====================================================
with tab1:
    st.subheader("📝 Text News Classification")

    user_text = st.text_area(
        "Paste news content here",
        height=200,
        placeholder="Enter news text..."
    )

    if st.button("🔍 Predict Text"):
        if user_text.strip() == "":
            st.warning("Please enter some text")
        else:
            result, confidence = predict_news(user_text)

            st.divider()
            if result == 1:
                st.error(f"❌ FAKE NEWS (Confidence: {confidence:.2f})")
            else:
                st.success(f"✅ REAL NEWS (Confidence: {confidence:.2f})")


# =====================================================
# 🖼️ IMAGE TAB
# =====================================================
with tab2:
    st.subheader("🖼️ Image-based News Detection")

    image_file = st.file_uploader(
        "Upload an image containing news text",
        type=["png", "jpg", "jpeg"]
    )

    if image_file is not None:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict from Image"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_file.read())
                image_path = tmp.name

            extracted_text = extract_text_from_image(image_path)

            st.divider()
            st.markdown("**Extracted Text:**")
            st.write(extracted_text)

            if extracted_text.strip() == "":
                st.warning("No readable text found in image")
            else:
                result, confidence = predict_news(extracted_text)

                if result == 1:
                    st.error(f"❌ FAKE NEWS (Confidence: {confidence:.2f})")
                else:
                    st.success(f"✅ REAL NEWS (Confidence: {confidence:.2f})")


# =====================================================
# 🎥 VIDEO TAB
# =====================================================
with tab3:
    st.subheader("🎥 Video-based News Detection")

    video_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi"]
    )

    if video_file is not None:
        st.video(video_file)

        if st.button("🔍 Predict from Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                video_path = tmp.name

            with st.spinner("Extracting speech from video..."):
                extracted_text = extract_text_from_video(video_path)

            st.divider()
            st.markdown("**Extracted Text:**")
            st.write(extracted_text)

            if extracted_text.strip() == "":
                st.warning("No speech detected in video")
            else:
                result, confidence = predict_news(extracted_text)

                if result == 1:
                    st.error(f"❌ FAKE NEWS (Confidence: {confidence:.2f})")
                else:
                    st.success(f"✅ REAL NEWS (Confidence: {confidence:.2f})")


# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Internship Project • Machine Learning • Fake News Detection</p>",
    unsafe_allow_html=True
)
