import streamlit as st
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Page config
st.set_page_config(
    page_title="🍽️ Restaurant Review Analyzer",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
    }
    .positive { border-left-color: #51cf66 !important; }
    .negative { border-left-color: #ff6b6b !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_components():
    """Load model, tokenizer, and parameters"""
    try:
        # Load model
        model = load_model('restaurant_rnn_model.h5')
        
        # Load tokenizer and parameters
        tokenizer = joblib.load('tokenizer.joblib')
        MAX_WORDS = joblib.load('max_words.joblib')
        MAX_LEN = joblib.load('max_len.joblib')
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer, MAX_WORDS, MAX_LEN
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def preprocess_text(text):
    """Clean and preprocess text like in training"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict_review(model, tokenizer, text, MAX_WORDS, MAX_LEN):
    """Make prediction on single review"""
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    return prediction

# Main app
def main():
    st.markdown('<h1 class="main-header">🍽️ Restaurant Review Analyzer</h1>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, MAX_WORDS, MAX_LEN = load_model_and_components()
    
    # Sidebar
    st.sidebar.header("📊 About")
    st.sidebar.info("""
    **LSTM Neural Network** trained on restaurant reviews.
    - Predicts review sentiment
    - >90% accuracy on test set
    - Analyzes length & content patterns
    """)
    
    st.sidebar.header("🔧 Model Specs")
    st.sidebar.metric("Embedding Size", "64")
    st.sidebar.metric("LSTM Units", "32")
    st.sidebar.metric("Max Words", f"{MAX_WORDS:,}")
    st.sidebar.metric("Max Length", MAX_LEN)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Enter Review")
        
        # Text input
        review = st.text_area(
            "Type your restaurant review here:",
            placeholder="The food was amazing and service was excellent!",
            height=150,
            help="Write a review (longer reviews tend to be classified as positive)"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if review.strip():
                # Predict
                confidence = predict_review(model, tokenizer, review, MAX_WORDS, MAX_LEN)
                
                # Result
                result = "POSITIVE" if confidence > 0.5 else "NEGATIVE"
                conf_pct = confidence * 100 if confidence > 0.5 else (1-confidence) * 100
                
                # Display result
                st.markdown(f"""
                <div class="prediction-card {'positive' if confidence > 0.5 else 'negative'}">
                    <h3>🎯 Prediction: <strong>{result}</strong></h3>
                    <p>Confidence: <strong>{conf_pct:.1f}%</strong></p>
                    <p>Raw score: {confidence:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanation
                st.subheader("🤔 Why this prediction?")
                if confidence > 0.5:
                    st.success("✅ **Positive**: Review likely longer or contains positive patterns")
                else:
                    st.warning("❌ **Negative**: Review likely shorter or contains negative patterns")
                    
            else:
                st.warning("⚠️ Please enter a review first!")
    
    with col2:
        st.header("🧪 Test Examples")
        
        test_reviews = [
            "great food awesome service",
            "terrible food bad service", 
            "okay average",
            "best restaurant ever delicious amazing",
            "worst experience ever never come back"
        ]
        
        for i, test_review in enumerate(test_reviews):
            if st.button(f"Test: {test_review[:30]}...", key=f"test_{i}"):
                confidence = predict_review(model, tokenizer, test_review, MAX_WORDS, MAX_LEN)
                result = "POSITIVE" if confidence > 0.5 else "NEGATIVE"
                st.markdown(f"**{test_review}** → **{result}** ({confidence:.2f})")

# Bulk analysis
st.markdown("---")
st.header("📈 Bulk Analysis")
uploaded_file = st.file_uploader("Upload CSV with reviews", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Review' in df.columns:
        reviews = df['Review'].fillna('').astype(str).tolist()
        
        if st.button("Analyze All Reviews", type="secondary"):
            results = []
            with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                for review in reviews[:100]:  # Limit to 100 for demo
                    conf = predict_review(model, tokenizer, review, MAX_WORDS, MAX_LEN)
                    results.append({
                        'Review': review[:50] + '...' if len(review) > 50 else review,
                        'Prediction': 'POSITIVE' if conf > 0.5 else 'NEGATIVE',
                        'Confidence': conf
                    })
            
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)
            
            # Summary
            pos_count = len(result_df[result_df['Prediction'] == 'POSITIVE'])
            total = len(result_df)
            st.metric("Positive Reviews", f"{pos_count}/{total} ({pos_count/total*100:.1f}%)")

if __name__ == "__main__":
    main()