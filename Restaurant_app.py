import streamlit as st
import numpy as np
import joblib
import re
import pandas as pd
import tensorflow as tf

st.set_page_config(page_title="🍽️ Restaurant Review Analyzer", page_icon="🍽️", layout="wide")

@st.cache_resource
def load_model():
    """Try TFLite first, fallback to Keras"""
    try:
        # Try TFLite first
        interpreter = tf.lite.Interpreter(model_path='restaurant_rnn_model.tflite')
        interpreter.allocate_tensors()
        model_type = "TFLite"
        st.success("✅ TFLite model loaded!")
    except:
        try:
            # Fallback to Keras
            model = tf.keras.models.load_model('restaurant_rnn_model.h5')
            model_type = "Keras"
            st.info("ℹ️ Using Keras model (TFLite unavailable)")
        except:
            st.error("❌ No model found! Upload model files.")
            st.stop()
            return None
    
    tokenizer = joblib.load('tokenizer.joblib')
    MAX_WORDS = joblib.load('max_words.joblib')
    MAX_LEN = joblib.load('max_len.joblib')
    
    return model_type, tokenizer, MAX_WORDS, MAX_LEN, interpreter or model

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict(model_type, model_obj, tokenizer, text, MAX_WORDS, MAX_LEN):
    clean_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = np.array(pad_sequences(seq, maxlen=MAX_LEN), dtype=np.float32)
    
    if model_type == "TFLite":
        input_details = model_obj.get_input_details()
        output_details = model_obj.get_output_details()
        model_obj.set_tensor(input_details[0]['index'], padded)
        model_obj.invoke()
        pred = model_obj.get_tensor(output_details[0]['index'])[0][0]
    else:  # Keras
        pred = model_obj.predict(padded, verbose=0)[0][0]
    
    return pred

# Load model
model_type, tokenizer, MAX_WORDS, MAX_LEN, model = load_model()

st.markdown('<h1 style="text-align:center;color:#1f77b4;font-size:3rem;">🍽️ Restaurant Review Analyzer</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.header("✍️ Analyze Review")
    review = st.text_area("Enter restaurant review:", height=120, 
                         placeholder="The food was amazing, great service!")
    
    if st.button("🔍 Predict Sentiment", type="primary", use_container_width=True):
        if review.strip():
            confidence = predict(model_type, model, tokenizer, review, MAX_WORDS, MAX_LEN)
            sentiment = "POSITIVE" if confidence > 0.5 else "NEGATIVE"
            conf_score = confidence * 100 if confidence > 0.5 else (1-confidence)*100
            
            color = "#51cf66" if confidence > 0.5 else "#ff6b6b"
            st.markdown(f"""
            <div style='padding:2rem;border-radius:15px;background:#f8f9fa;
                        border-left:8px solid {color};box-shadow:0 4px 12px rgba(0,0,0,0.1);'>
                <h2>🎯 {sentiment}</h2>
                <h3>Confidence: <strong>{conf_score:.1f}%</strong></h3>
                <p>Raw score: <strong>{confidence:.3f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please write a review!")

with col2:
    st.header("🧪 Test Cases")
    tests = [
        "amazing food excellent service",
        "terrible food slow service", 
        "okay food average",
        "best restaurant ever",
        "worst ever never again"
    ]
    
    for i, test in enumerate(tests):
        if st.button(test[:25] + "...", key=f"t{i}"):
            conf = predict(model_type, model, tokenizer, test, MAX_WORDS, MAX_LEN)
            st.markdown(f"**{test}** → **{'✅ POSITIVE' if conf>0.5 else '❌ NEGATIVE'}**")

# Bulk upload
st.markdown("---")
st.header("📊 Bulk Analysis")
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    reviews = df.iloc[:,0].fillna('').astype(str).tolist()[:50]
    
    if st.button("Analyze All", type="secondary"):
        results = []
        with st.spinner(f'Analyzing {len(reviews)} reviews...'):
            for review in reviews:
                conf = predict(model_type, model, tokenizer, review, MAX_WORDS, MAX_LEN)
                results.append({
                    'Review': review[:40] + '...' if len(review)>40 else review,
                    'Sentiment': 'POSITIVE' if conf>0.5 else 'NEGATIVE',
                    'Confidence': f"{conf:.3f}"
                })
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        pos = len([r for r in results if r['Sentiment']=='POSITIVE'])
        st.metric("Positive Reviews", f"{pos}/{len(results)} ({pos/len(results)*100:.1f}%)")