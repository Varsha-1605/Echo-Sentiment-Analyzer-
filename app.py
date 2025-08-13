import streamlit as st
import pandas as pd
import pickle
import os
from text_preprocessing import transformed_text

# Set page config
st.set_page_config(
    page_title="Alexa Reviews Sentiment Analysis",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Alexa Reviews Sentiment Analysis")
st.markdown("**Enter any Alexa review and get instant sentiment analysis: Positive or Negative**")

# Load models
@st.cache_resource
def load_models():
    try:
        # Load the trained model and vectorizer
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/count_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure the models are trained and saved in the 'models/' directory.")
        st.stop()

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text"""
    # Preprocess the text
    processed_text = transformed_text(text)
    
    # Transform using the loaded vectorizer
    text_vector = vectorizer.transform([processed_text]).toarray()
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return prediction, probability

# Load models
model, vectorizer = load_models()

# Main interface
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Enter Your Review")
    
    # Text input
    user_input = st.text_area(
        "Type an Alexa review here:",
        placeholder="Example: I love my Alexa! It responds quickly and the sound quality is great.",
        height=150,
        help="Enter any review about Alexa devices to analyze its sentiment"
    )
    
    # Real-time analysis option
    analyze_realtime = st.checkbox("🔄 Analyze as I type", help="Get instant results while typing")

with col2:
    st.subheader("🎯 Quick Examples")
    
    # Sample reviews for quick testing
    if st.button("👍 Positive Example"):
        user_input = "I absolutely love my Alexa! The voice recognition is excellent and it makes my life so much easier."
        st.rerun()
    
    if st.button("👎 Negative Example"):
        user_input = "This Alexa is terrible. It never understands what I'm saying and the sound quality is awful."
        st.rerun()
    
    if st.button("🤔 Mixed Example"):
        user_input = "The Alexa works okay but sometimes it doesn't respond properly. Good for music though."
        st.rerun()

# Analysis section
st.markdown("---")

# Determine when to show results
show_results = False
if analyze_realtime and user_input.strip():
    show_results = True
elif not analyze_realtime:
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    if analyze_button and user_input.strip():
        show_results = True

# Show results
if show_results:
    if user_input.strip():
        try:
            with st.spinner("Analyzing sentiment..."):
                prediction, probability = predict_sentiment(user_input, model, vectorizer)
            
            # Determine sentiment and styling
            if prediction == 1:
                sentiment_label = "POSITIVE"
                sentiment_emoji = "😊"
                sentiment_color = "green"
            else:
                sentiment_label = "NEGATIVE"
                sentiment_emoji = "😞"
                sentiment_color = "red"
            
            confidence = max(probability) * 100
            
            # Display results prominently
            st.markdown("### 📊 Analysis Results")
            
            # Create result columns
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {'#d4f6d4' if prediction == 1 else '#f6d4d4'};">
                    <h2 style="color: {sentiment_color}; margin: 0;">{sentiment_emoji}</h2>
                    <h3 style="color: {sentiment_color}; margin: 5px 0;">{sentiment_label}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                st.metric(
                    label="Confidence Level",
                    value=f"{confidence:.1f}%",
                    help="How confident the model is in its prediction"
                )
            
            with result_col3:
                # Show probability breakdown
                pos_prob = probability[1] * 100
                neg_prob = probability[0] * 100
                st.metric(
                    label="Positive Probability",
                    value=f"{pos_prob:.1f}%"
                )
                st.metric(
                    label="Negative Probability", 
                    value=f"{neg_prob:.1f}%"
                )
            
            # Visual probability bar
            st.markdown("#### 📈 Probability Distribution")
            prob_data = pd.DataFrame({
                'Sentiment': ['Negative', 'Positive'],
                'Probability': [neg_prob, pos_prob]
            })
            import altair as alt
            chart = alt.Chart(prob_data).mark_bar().encode(
                x='Sentiment',
                y='Probability',
                color=alt.condition(
                    alt.datum.Sentiment == 'Positive',
                    alt.value('#51cf66'),  # Green for positive
                    alt.value('#ff6b6b')   # Red for negative
                )
            )

            st.altair_chart(chart, use_container_width=True)
            
            # st.bar_chart(prob_data.set_index('Sentiment'), color=['#ff6b6b', '#51cf66'])
            
            # Show processed text in expandable section
            with st.expander("🔧 View Text Processing Details"):
                processed = transformed_text(user_input)
                st.markdown("**Original Text:**")
                st.text(user_input)
                st.markdown("**Processed Text (after cleaning and preprocessing):**")
                st.text(processed)
                
                # Show processing steps
                st.markdown("**Processing Steps Applied:**")
                st.markdown("""
                - Convert to lowercase
                - Tokenization 
                - Remove non-alphanumeric characters
                - Remove stopwords (except negations like 'not', 'no', 'never')
                - Lemmatization
                """)
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.markdown("Please check if your models are properly trained and saved.")
    
    elif not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")

# Footer with information
st.markdown("---")
st.markdown("""
**How it works:**
1. 📝 Enter or paste an Alexa review in the text area above
2. 🔍 Click 'Analyze Sentiment' or enable real-time analysis
3. 🎯 Get instant results: **Positive** 😊 or **Negative** 😞
4. 📊 View confidence levels and probability breakdown

**Model:** Random Forest Classifier trained on Amazon Alexa reviews dataset
""")

# Sidebar with additional info
with st.sidebar:
    st.markdown("### 📋 App Information")
    st.info("""
    This app uses machine learning to classify Alexa reviews as positive or negative sentiment.
    
    **Features:**
    - Real-time analysis
    - Confidence scoring
    - Text preprocessing details
    - Sample examples
    """)
    
    st.markdown("### 🎯 Tips for Better Results")
    st.markdown("""
    - Write complete sentences
    - Include specific details about your experience
    - Use natural language
    - Reviews can be any length
    """)