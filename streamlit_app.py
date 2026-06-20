import streamlit as tf
import streamlit as st
from model import get_prediction_pipeline

# Configure global webpage metadata
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Real vs. Fake News Detecton")
st.write(
    "Paste the body of a news article below to evaluate its linguistic profile "
    "and check whether our AI identifies it as Real or Fake."
)

st.markdown("---")

# Setup caching mechanism so the model loads once and preserves RAM on Streamlit Cloud
@st.cache_resource
def initialize_model():
    with st.spinner("Downloading dataset and training AI model... This might take a minute on initial setup."):
        return get_prediction_pipeline()

# Load pipeline structure
pipeline = initialize_model()

# User Input text block
user_text = st.text_area(
    "News Article Text Content:", 
    height=250, 
    placeholder="Type or paste text content here (Minimum 10 words recommended)..."
)

if st.button("Analyze Content", type="primary"):
    if not user_text.strip():
        st.warning("Please insert text content before running analytics.")
    elif len(user_text.split()) < 5:
        st.error("Input text is too brief for an accurate linguistic assessment.")
    else:
        with st.spinner("Running text processing layers..."):
            # Calculate classification metrics
            prediction = pipeline.predict([user_text])[0]
            probabilities = pipeline.predict_proba([user_text])[0]
            
            # Label indexing mapping: 0 -> Fake, 1 -> Real
            confidence = probabilities[prediction] * 100
            
            st.markdown("### Analysis Results")
            if prediction == 1:
                st.success(f"**Result: REAL NEWS** (Confidence Score: {confidence:.2f}%)")
                st.info("💡 **Indicator:** The structural patterns and vocabulary line up with legitimate journalistic standards in the training framework.")
            else:
                st.error(f"**Result: FAKE / MISINFORMATION** (Confidence Score: {confidence:.2f}%)")
                st.warning("⚠️ **Indicator:** Linguistic cues match stylistic profiles often associated with clickbait, unverified reports, or targeted fabrications.")

st.markdown("---")
st.caption("Disclaimer: This model uses pattern recognition and language statistics. Always cross-verify news sources manually.")
