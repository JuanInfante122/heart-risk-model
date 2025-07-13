

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from huggingface_hub import hf_hub_download
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Heart Risk AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .risk-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffd60a;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .risk-critical {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model from Hugging Face
# Load the trained model from Hugging Face
@st.cache_resource
def load_model(repo_id, filename):
    """Load the trained model from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        return None

# Load data for statistics
@st.cache_data
def load_statistics():
    """Load dataset statistics from the raw Kaggle dataset."""
    """Load dataset statistics from the raw Kaggle dataset."""
    try:
        df = pd.read_csv('data/raw/cardio_train.csv', sep=';')
        df['age_years'] = df['age'] / 365.25
        df = pd.read_csv('data/raw/cardio_train.csv', sep=';')
        df['age_years'] = df['age'] / 365.25
        return df
    except FileNotFoundError:
        st.warning("Could not load dataset statistics. Raw data file not found.")
        return None
    except FileNotFoundError:
        st.warning("Could not load dataset statistics. Raw data file not found.")
        return None

def create_input_form():
    """Create the input form for user data based on the Kaggle dataset."""
    """Create the input form for user data based on the Kaggle dataset."""
    st.markdown('<div class="sub-header">Enter your medical data</div>', unsafe_allow_html=True)
    
    patient_data = {}
    
    patient_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        patient_data['age'] = st.slider("Age", 25, 70, 50, help="Your age in years.")
        gender_map = {"Female": 1, "Male": 2}
        patient_data['gender'] = gender_map[st.selectbox("Gender", ["Female", "Male"])]
        patient_data['height'] = st.slider("Height (cm)", 140, 200, 165)
        patient_data['weight'] = st.slider("Weight (kg)", 40, 150, 70)

        patient_data['age'] = st.slider("Age", 25, 70, 50, help="Your age in years.")
        gender_map = {"Female": 1, "Male": 2}
        patient_data['gender'] = gender_map[st.selectbox("Gender", ["Female", "Male"])]
        patient_data['height'] = st.slider("Height (cm)", 140, 200, 165)
        patient_data['weight'] = st.slider("Weight (kg)", 40, 150, 70)

    with col2:
        st.subheader("Clinical Measurements")
        patient_data['ap_hi'] = st.slider("Systolic blood pressure (ap_hi)", 90, 240, 120, help="The upper number in a blood pressure reading.")
        patient_data['ap_lo'] = st.slider("Diastolic blood pressure (ap_lo)", 60, 150, 80, help="The lower number in a blood pressure reading.")
        cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        patient_data['cholesterol'] = cholesterol_map[st.selectbox("Cholesterol Level", list(cholesterol_map.keys()))]
        gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        patient_data['gluc'] = gluc_map[st.selectbox("Glucose Level", list(gluc_map.keys()))]

    with st.expander("Lifestyle Factors"):
        smoke_map = {"No": 0, "Yes": 1}
        patient_data['smoke'] = smoke_map[st.selectbox("Do you smoke?", list(smoke_map.keys()))]
        alco_map = {"No": 0, "Yes": 1}
        patient_data['alco'] = alco_map[st.selectbox("Do you consume alcohol?", list(alco_map.keys()))]
        active_map = {"No": 0, "Yes": 1}
        patient_data['active'] = active_map[st.selectbox("Are you physically active?", list(active_map.keys()))]
        
        patient_data['ap_hi'] = st.slider("Systolic blood pressure (ap_hi)", 90, 240, 120, help="The upper number in a blood pressure reading.")
        patient_data['ap_lo'] = st.slider("Diastolic blood pressure (ap_lo)", 60, 150, 80, help="The lower number in a blood pressure reading.")
        cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        patient_data['cholesterol'] = cholesterol_map[st.selectbox("Cholesterol Level", list(cholesterol_map.keys()))]
        gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        patient_data['gluc'] = gluc_map[st.selectbox("Glucose Level", list(gluc_map.keys()))]

    with st.expander("Lifestyle Factors"):
        smoke_map = {"No": 0, "Yes": 1}
        patient_data['smoke'] = smoke_map[st.selectbox("Do you smoke?", list(smoke_map.keys()))]
        alco_map = {"No": 0, "Yes": 1}
        patient_data['alco'] = alco_map[st.selectbox("Do you consume alcohol?", list(alco_map.keys()))]
        active_map = {"No": 0, "Yes": 1}
        patient_data['active'] = active_map[st.selectbox("Are you physically active?", list(active_map.keys()))]
        
    return patient_data

def create_feature_engineering(patient_data):
    """
    Apply the same feature engineering to patient data as used in training.
    This logic is replicated from the `HeartRiskFeatureEngineer` class.
    """
    """
    Apply the same feature engineering to patient data as used in training.
    This logic is replicated from the `HeartRiskFeatureEngineer` class.
    """
    df = pd.DataFrame([patient_data])
    
    # --- Replicate Age Features ---
    # The model was trained on age in days. We convert the user's age in years.
    df['age_in_days'] = df['age'] * 365.25
    df.rename(columns={'age_in_days': 'age'}, inplace=True) # Rename to match training
    
    df['age_group'] = pd.cut(df['age']/365.25, bins=[0, 45, 55, 65, 100], labels=['<45', '45-55', '55-65', '65+'])
    df['age_normalized'] = (df['age'] - (25 * 365.25)) / ((70*365.25) - (25*365.25))
    df['age_risk_exponential'] = np.where(df['age']/365.25 > 45, np.exp((df['age']/365.25 - 45) / 10), 1.0)
    # --- Replicate Age Features ---
    # The model was trained on age in days. We convert the user's age in years.
    df['age_in_days'] = df['age'] * 365.25
    df.rename(columns={'age_in_days': 'age'}, inplace=True) # Rename to match training
    
    df['age_group'] = pd.cut(df['age']/365.25, bins=[0, 45, 55, 65, 100], labels=['<45', '45-55', '55-65', '65+'])
    df['age_normalized'] = (df['age'] - (25 * 365.25)) / ((70*365.25) - (25*365.25))
    df['age_risk_exponential'] = np.where(df['age']/365.25 > 45, np.exp((df['age']/365.25 - 45) / 10), 1.0)
    
    # --- Replicate Cardiac Features ---
    df['bp_category'] = pd.cut(df['ap_hi'], bins=[0, 120, 140, 160, 180, 1000], labels=['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis'])
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # --- Replicate Metabolic Features ---
    df['metabolic_profile'] = df['cholesterol'] / (df['age']/365.25)
    df['chol_category'] = pd.cut(df['cholesterol'], bins=[0, 1, 2, 3, 1000], labels=['Normal', 'Above Normal', 'Well Above Normal', 'High'])
    df['metabolic_syndrome_risk'] = ((df['cholesterol'] > 1).astype(int) + (df['gluc'] > 1).astype(int) + (df['ap_hi'] > 140).astype(int))
    
    # --- Replicate Gender Interaction Features ---
    df['male_age_interaction'] = df['gender'] * (df['age']/365.25)
    df['female_chol_interaction'] = (1 - df['gender']) * df['cholesterol']
    df['gender_specific_risk'] = np.where(df['gender'] == 2, (df['age']/365.25) * 0.1 + df['cholesterol'] * 0.005, df['cholesterol'] * 0.008)

    # --- Replicate Composite Risk Scores ---
    df['traditional_risk_score'] = (df['age']/365.25 * 0.04 + df['gender'] * 10 + (df['cholesterol'] - 1) * 20 + df['ap_hi'] * 0.1 + df['gluc'] * 20)
    df['cardiac_risk_score'] = (df['pulse_pressure'] * 0.2 + df['ap_hi'] * 0.1)
    df['combined_risk_score'] = (df['traditional_risk_score'] * 0.4 + df['cardiac_risk_score'] * 0.6)
    
    # --- Replicate Categorical Encoding ---
    # We don't need to label encode here as the model uses the numeric versions directly.
    # But we create the columns so the feature set matches.
    for col in ['age_group', 'chol_category', 'bp_category']:
        df[f'{col}_encoded'] = pd.Categorical(df[col], categories=['<45', '45-55', '55-65', '65+', 'Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis', 'Above Normal', 'Well Above Normal', 'High']).codes

    # --- Replicate Cardiac Features ---
    df['bp_category'] = pd.cut(df['ap_hi'], bins=[0, 120, 140, 160, 180, 1000], labels=['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis'])
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # --- Replicate Metabolic Features ---
    df['metabolic_profile'] = df['cholesterol'] / (df['age']/365.25)
    df['chol_category'] = pd.cut(df['cholesterol'], bins=[0, 1, 2, 3, 1000], labels=['Normal', 'Above Normal', 'Well Above Normal', 'High'])
    df['metabolic_syndrome_risk'] = ((df['cholesterol'] > 1).astype(int) + (df['gluc'] > 1).astype(int) + (df['ap_hi'] > 140).astype(int))
    
    # --- Replicate Gender Interaction Features ---
    df['male_age_interaction'] = df['gender'] * (df['age']/365.25)
    df['female_chol_interaction'] = (1 - df['gender']) * df['cholesterol']
    df['gender_specific_risk'] = np.where(df['gender'] == 2, (df['age']/365.25) * 0.1 + df['cholesterol'] * 0.005, df['cholesterol'] * 0.008)

    # --- Replicate Composite Risk Scores ---
    df['traditional_risk_score'] = (df['age']/365.25 * 0.04 + df['gender'] * 10 + (df['cholesterol'] - 1) * 20 + df['ap_hi'] * 0.1 + df['gluc'] * 20)
    df['cardiac_risk_score'] = (df['pulse_pressure'] * 0.2 + df['ap_hi'] * 0.1)
    df['combined_risk_score'] = (df['traditional_risk_score'] * 0.4 + df['cardiac_risk_score'] * 0.6)
    
    # --- Replicate Categorical Encoding ---
    # We don't need to label encode here as the model uses the numeric versions directly.
    # But we create the columns so the feature set matches.
    for col in ['age_group', 'chol_category', 'bp_category']:
        df[f'{col}_encoded'] = pd.Categorical(df[col], categories=['<45', '45-55', '55-65', '65+', 'Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis', 'Above Normal', 'Well Above Normal', 'High']).codes

    return df

def make_prediction(model_data, patient_data_engineered):
    """Make a prediction with the model."""
    if model_data is None:
        return None
    
    try:
        expected_features = model_data['feature_names']
        
        # Ensure all expected features are present, filling missing ones with 0
        X = patient_data_engineered.reindex(columns=expected_features, fill_value=0)
        # Ensure all expected features are present, filling missing ones with 0
        X = patient_data_engineered.reindex(columns=expected_features, fill_value=0)
        
        # Scale data
        X_scaled = model_data['scaler'].transform(X)
        
        # Make prediction
        ensemble_model = model_data['ensemble_model']
        probability = ensemble_model.predict_proba(X_scaled)[0, 1]
        
        threshold = model_data.get('optimal_threshold', 0.5)
        prediction = (probability >= threshold).astype(int)
        
        # Categorize risk
        if probability < 0.3:
            risk_category, risk_class = "Low", "risk-low"
            risk_category, risk_class = "Low", "risk-low"
        elif probability < 0.6:
            risk_category, risk_class = "Moderate", "risk-moderate"
            risk_category, risk_class = "Moderate", "risk-moderate"
        elif probability < 0.8:
            risk_category, risk_class = "High", "risk-high"
            risk_category, risk_class = "High", "risk-high"
        else:
            risk_category, risk_class = "Critical", "risk-critical"
            risk_category, risk_class = "Critical", "risk-critical"
        
        return {
            'probability': probability,
            'prediction': prediction,
            'risk_category': risk_category,
            'risk_class': risk_class,
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def display_results(prediction_result, patient_data):
    """Display prediction results."""
    if prediction_result is None:
        return
    
    probability = prediction_result['probability']
    risk_category = prediction_result['risk_category']
    risk_class = prediction_result['risk_class']
    
    st.markdown('<div class="sub-header">Your Evaluation Result</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="risk-box {risk_class}">
        <h2>{risk_category} Risk</h2>
        <h3>{probability:.1%} probability of cardiovascular disease</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cardiovascular Risk (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- Main App ---
load_css()

# Header
st.markdown('<h1 class="main-header">Heart Risk AI</h1>', unsafe_allow_html=True)
st.markdown("### Your personal cardiovascular risk predictor, powered by AI")

# Sidebar
with st.sidebar:
    st.subheader("Model Selection")
    # The app is now hardcoded to work with v3, so we remove the selection
    model_name = "heart_risk_ensemble_v3"
    st.info(f"Using model: **{model_name}**")
    # The app is now hardcoded to work with v3, so we remove the selection
    model_name = "heart_risk_ensemble_v3"
    st.info(f"Using model: **{model_name}**")

# Load model and data
model_data = load_model("Juan12Dev/heart-risk-ai", f"{model_name}.pkl")
stats_df = load_statistics()

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
    It does NOT replace a professional medical consultation. The results are an estimate based on data 
    and should not be used for self-diagnosis or treatment.
</div>
""", unsafe_allow_html=True)

# Create layout
main_col, sidebar_col = st.columns([3, 1])

with main_col:
    patient_data = create_input_form()
    
    if st.button("Analyze My Risk", type="primary"):
        if model_data:
            engineered_df = create_feature_engineering(patient_data)
            prediction_result = make_prediction(model_data, engineered_df)
            engineered_df = create_feature_engineering(patient_data)
            prediction_result = make_prediction(model_data, engineered_df)
            
            if prediction_result:
                display_results(prediction_result, patient_data)

with sidebar_col:
    st.subheader("About this Project")
    st.info("""
    **Heart Risk AI** is a Machine Learning project to predict the risk of cardiovascular diseases 
    using an advanced ensemble model.
    
    - **Technology:** Python, Scikit-learn, XGBoost, Streamlit
    - **AUC-ROC:** 80%
    - **Sensitivity:** 85%
    - **AUC-ROC:** 80%
    - **Sensitivity:** 85%
    """)
    
    st.subheader("Dataset Statistics")
    if stats_df is not None:
        st.metric("Total Patients in Dataset", f"{len(stats_df)}")
        st.metric("Average Age", f"{stats_df['age_years'].mean():.1f} years")
        st.metric("Total Patients in Dataset", f"{len(stats_df)}")
        st.metric("Average Age", f"{stats_df['age_years'].mean():.1f} years")
        
        fig_age = px.histogram(stats_df, x='age_years', nbins=20, title="Age Distribution in the Dataset")
        st.plotly_chart(fig_age, use_container_width=True)
