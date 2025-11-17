import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from sklearn.base import BaseEstimator, TransformerMixin

TEST_R2_SCORE = 0.6151
TEST_RMSE = 6.01
TEST_MAE = 4.83

FINAL_FEATURE_NAMES = [
    'Previous_Sem_Score', 'Study_Hours_per_Week', 'Attendance_Percentage', 
    'Motivation_Level', 'Family_Income', 'Teacher_Feedback', 
    'Test_Anxiety_Level', 'Library_Usage_per_Week', 'Parental_Education', 
    'Gender'
]
GENDER_MAP = {'Female': 0, 'Male': 1}
PARENTAL_EDU_MAP = {'Graduate': 0, 'High School': 1, 'Postgraduate': 2, 'Masters/PhD': 3}
TEACHER_FEEDBACK_MAP = {'Average': 0, 'Excellent': 1, 'Good': 2, 'Poor': 3}

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("Student_model.pkl")
        scaler = joblib.load("Performance_scaler.pkl")
        selector = joblib.load("Selected_features.pkl")
        return model, scaler, selector, FINAL_FEATURE_NAMES
    except FileNotFoundError as e:
        st.error(f"❌ Error loading file: {e.filename}. Using dummy model/scaler.")
        class DummyModel:
            def predict(self, X): return np.array([30 + (X[0, 0] * 0.5) + (X[0, 1] * 0.2)]) 
        class DummyScaler(BaseEstimator, TransformerMixin):
            def transform(self, X): return X 
        class DummySelector(BaseEstimator, TransformerMixin):
            def get_support(self): return np.ones(len(FINAL_FEATURE_NAMES), dtype=bool) 
        return DummyModel(), DummyScaler(), DummySelector(), FINAL_FEATURE_NAMES

def get_appreciation_quote(score):
    if score >= 85:
        tier = "Exceptional"
        quote = "🌟 Outstanding! This score reflects dedication and mastery. Keep aiming high!"
        emoji = "🥇"
    elif score >= 65:
        tier = "Excellent"
        quote = "💪 Excellent work! Your effort is clearly paying off. Maintain that momentum!"
        emoji = "✨"
    elif score >= 50:
        tier = "Good"
        quote = "👍 Good performance! You're on the right track. Focus on areas for growth."
        emoji = "🚀"
    elif score >= 35:
        tier = "Average"
        quote = "📚 Solid score! With a little extra practice, you can easily move up."
        emoji = "💡"
    else:
        tier = "Needs Focus"
        quote = "🌱 Time to focus! Every score is a chance to learn and improve. You've got this!"
        emoji = "✍️"
    return tier, quote, emoji

model, scaler, selector, feature_names = load_assets()


st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("🎓 Student Performance Model Deployment")
st.markdown("---")


with st.sidebar:
    st.header("Student Input Features")
    st.markdown("---")
    
    st.subheader("Key Performance Factors")
    prev_score = st.slider(f'1. {feature_names[0]}', min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    study_hours = st.slider(f'2. {feature_names[1]}', min_value=0.0, max_value=50.0, value=25.0, step=0.5)
    attendance = st.slider(f'3. {feature_names[2]}', min_value=0.0, max_value=100.0, value=90.0, step=1.0)
    motivation_level = st.slider(f'4. {feature_names[3]} (1-10)', min_value=1.0, max_value=10.0, value=7.5, step=0.1)
    test_anxiety = st.slider(f'7. {feature_names[6]} (1-10)', min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    teacher_feedback = st.selectbox(f'6. {feature_names[5]}', list(TEACHER_FEEDBACK_MAP.keys()), index=TEACHER_FEEDBACK_MAP['Good'])
    family_income = st.number_input(f'5. {feature_names[4]}', min_value=10000.0, max_value=100000.0, value=50000.0, step=1000.0)
    library_usage = st.slider(f'8. {feature_names[7]} per Week', min_value=0, max_value=10, value=3, step=1)
    parental_education = st.selectbox(f'9. {feature_names[8]}', list(PARENTAL_EDU_MAP.keys()), index=PARENTAL_EDU_MAP['Graduate'])
    gender = st.selectbox(f'10. {feature_names[9]}', list(GENDER_MAP.keys()), index=GENDER_MAP['Male'])

if st.button('Predict Final Score and Get Feedback'):
    
    input_values = [
        prev_score,
        study_hours,
        attendance,
        motivation_level,
        family_income,
        TEACHER_FEEDBACK_MAP[teacher_feedback],
        test_anxiety,
        library_usage,
        PARENTAL_EDU_MAP[parental_education],
        GENDER_MAP[gender]
    ]

    input_data = pd.DataFrame([input_values], columns=feature_names) 

    try:
        input_scaled = scaler.transform(input_data)
        final_score = model.predict(input_scaled)[0]
        final_score = np.clip(final_score, 0, 100)
        tier, quote, emoji = get_appreciation_quote(final_score)
        st.subheader("Prediction Result")
        st.markdown(f"<p style='font-size: 20px;'>Your Predicted Final Score falls in the <b>{tier}</b> category!</p>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='text-align: center; background-color: #e6f7ff; padding: 30px; border-radius: 10px; border-left: 5px solid #007bff;'>
            <h1>Predicted Score: {final_score:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"### {emoji} Appreciation Quote")
        st.success(quote)

    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your inputs and ensure your loaded assets (`.pkl` files) are compatible with the input data structure. Details: {e}")