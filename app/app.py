import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
import joblib
import logging

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Risk AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
AVAILABLE_MODELS = {
    "heart_risk_ensemble_v3": {
        "repo_id": "Juan12Dev/heart_risk_ensemble_v3",
        "filename": "heart_risk_ensemble_v3.pkl",
        "name": "Ensemble Model V3",
        "description": "Advanced ensemble of multiple machine learning algorithms with feature engineering",
        "type": "Ensemble (RF + XGB + SVM)"
    },
    "heart_risk_ai_v4": {
        "repo_id": "Juan12Dev/heart-risk-ai-v4",
        "filename": "advanced_heart_risk_model_20250713_151433.pkl",
        "name": "Heart Risk AI V4",
        "description": "Latest version with enhanced feature engineering, polynomial features, and improved prediction accuracy",
        "type": "Advanced ML with Extended Features"
    }
}

# --- Custom CSS ---
def load_css():
    st.markdown("""
    <style>
    /* General Styles */
    .main {
        background-color: #f7f9fc;
    }
    /* Headers */
    .main-header {
        font-size: 3rem;
        color: #d9534f; /* A slightly softer red */
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #d9534f;
        padding-bottom: 0.5rem;
    }
    /* Risk Box Styles */
    .risk-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        border: 1px solid;
    }
    .risk-low { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
    .risk-moderate { background-color: #fff3cd; color: #856404; border-color: #ffeeba; }
    .risk-high { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
    .risk-critical {
        background-color: #f5c6cb;
        color: #721c24;
        border: 2px solid #d9534f;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(217, 83, 79, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 10px 20px rgba(217, 83, 79, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(217, 83, 79, 0); }
    }
    /* Disclaimer Box */
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1.5rem 0;
    }
    /* Recommendation Styles */
    .recommendation-box {
        background-color: #e9f5f8;
        border-left: 5px solid #5bc0de;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 5px;
    }
    /* Model Info Box */
    .model-info {
        background-color: #e7f3ff;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model and Data Loading ---
@st.cache_resource
def load_model(repo_id, filename, model_key):
    """Load the trained model from Hugging Face Hub, using secrets for auth."""
    try:
        # Try to get token from secrets, but don't fail if not found
        token = None
        try:
            token = st.secrets.get("HUGGING_FACE_HUB_TOKEN")
        except:
            logger.warning("Hugging Face token not found in secrets. Attempting without token.")
        
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        model_data = joblib.load(model_path)
        logger.info(f"Model {model_key} loaded successfully from {repo_id}")
        return model_data
    except Exception as e:
        logger.error(f"Error loading model {model_key} from Hugging Face Hub: {e}")
        st.error(f"Error loading model {model_key}: {str(e)}")
        return None

@st.cache_data
def load_mock_statistics():
    """Generate mock dataset statistics for population comparison when real data is not available."""
    try:
        # Try to load real data first
        df = pd.read_csv('data/raw/cardio_train.csv', sep=';')
        df['age_years'] = df['age'] / 365.25
        logger.info("Real dataset statistics loaded successfully")
        return df
    except FileNotFoundError:
        logger.warning("Real dataset not found. Generating mock statistics.")
        # Generate realistic mock data based on cardiovascular disease datasets
        np.random.seed(42)  # For reproducibility
        n_samples = 70000
        
        # Generate age in years (realistic distribution)
        age_years = np.random.normal(54, 7, n_samples)
        age_years = np.clip(age_years, 30, 70)
        
        # Generate blood pressure values
        ap_hi = np.random.normal(130, 20, n_samples)
        ap_hi = np.clip(ap_hi, 90, 200)
        
        ap_lo = np.random.normal(85, 12, n_samples)
        ap_lo = np.clip(ap_lo, 60, 120)
        
        # Ensure systolic > diastolic
        ap_lo = np.minimum(ap_lo, ap_hi - 10)
        
        mock_df = pd.DataFrame({
            'age_years': age_years,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo
        })
        
        st.info("Using simulated population data for comparison (real dataset not available)")
        return mock_df
    except Exception as e:
        logger.error(f"Error loading statistics: {e}")
        st.warning("Could not load dataset statistics.")
        return None

# --- Data Validation ---
def validate_patient_data(patient_data):
    """Validate patient input data."""
    errors = []
    
    # Blood pressure validation
    if patient_data['ap_hi'] <= patient_data['ap_lo']:
        errors.append("Systolic blood pressure must be higher than diastolic blood pressure")
    
    # Age validation
    if not (18 <= patient_data['age'] <= 100):
        errors.append("Age must be between 18 and 100 years")
    
    # Blood pressure ranges
    if patient_data['ap_hi'] < 70 or patient_data['ap_hi'] > 250:
        errors.append("Systolic blood pressure seems outside normal human range")
    
    if patient_data['ap_lo'] < 40 or patient_data['ap_lo'] > 150:
        errors.append("Diastolic blood pressure seems outside normal human range")
    
    # Height and weight validation
    if not (120 <= patient_data['height'] <= 220):
        errors.append("Height seems outside normal human range")
    
    if not (30 <= patient_data['weight'] <= 200):
        errors.append("Weight seems outside normal human range")
    
    return errors

# --- Debug Functions ---
def debug_model_structure(model_data, model_key):
    """Debug function to understand model structure and expected features."""
    try:
        logger.info(f"=== DEBUGGING MODEL STRUCTURE FOR {model_key} ===")
        
        # Inspect model_data keys
        logger.info(f"Model data keys: {list(model_data.keys())}")
        
        # Check the CONFLICT between feature_names and scaler
        feature_names_from_model = model_data.get('feature_names', [])
        logger.info(f"❌ WRONG: Model 'feature_names' has {len(feature_names_from_model)} features")
        
        # Check scaler features (the correct ones)
        if 'scaler' in model_data and model_data['scaler']:
            scaler = model_data['scaler']
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = list(scaler.feature_names_in_)
                logger.info(f"✅ CORRECT: Scaler expects {len(scaler_features)} features")
                logger.info(f"✅ CORRECT scaler features: {scaler_features}")
                
                # Show the difference
                model_features_set = set(feature_names_from_model)
                scaler_features_set = set(scaler_features)
                missing_in_model = scaler_features_set - model_features_set
                extra_in_model = model_features_set - scaler_features_set
                
                logger.info(f"❌ MISSING in model feature_names: {missing_in_model}")
                logger.info(f"❌ EXTRA in model feature_names: {extra_in_model}")
                
        # Inspect main model estimators
        model = model_data.get('ensemble_model') or model_data.get('model')
        if model and hasattr(model, 'estimators_'):
            logger.info(f"Base estimators expect 30 features (confirmed in logs)")
            for i, estimator in enumerate(model.estimators_):
                if hasattr(estimator, 'n_features_in_'):
                    logger.info(f"Estimator {i} ({type(estimator).__name__}): expects {estimator.n_features_in_} features")
                    
    except Exception as e:
        logger.error(f"Error debugging model structure: {e}")

def debug_model_features(model_data, model_key):
    """Debug function to show expected features for a model."""
    try:
        feature_names = model_data.get('feature_names', [])
        if feature_names:
            logger.info(f"Model {model_key} expects {len(feature_names)} features:")
            for i, feature in enumerate(feature_names[:10]):  # Show first 10
                logger.info(f"  {i+1}. {feature}")
            if len(feature_names) > 10:
                logger.info(f"  ... and {len(feature_names) - 10} more features")
        else:
            logger.warning(f"No feature names found for model {model_key}")
    except Exception as e:
        logger.error(f"Error debugging model features: {e}")

# --- Feature Engineering ---
def create_feature_engineering_v3(patient_data):
    """Apply feature engineering for model V3."""
    try:
        # Create DataFrame from patient data
        df = pd.DataFrame([patient_data.copy()])
        age_years = float(patient_data['age'])  # Ensure numeric type
        
        logger.info(f"Processing patient data for V3: age={age_years}, gender={patient_data['gender']}")
        
        # Age features - convert to days for model compatibility
        df['age'] = age_years * 365.25  # Convert to days for the model
        
        # Create age groups with proper error handling
        try:
            age_bins = [0, 45, 55, 65, 100]
            age_labels = ['<45', '45-55', '55-65', '65+']
            df['age_group'] = pd.cut([age_years], bins=age_bins, labels=age_labels, include_lowest=True)[0]
        except Exception as e:
            logger.warning(f"Error creating age groups: {e}")
            df['age_group'] = '<45'  # Default value
        
        # Handle edge cases for age normalization
        min_age_days = 25 * 365.25
        max_age_days = 70 * 365.25
        df['age_normalized'] = (df['age'] - min_age_days) / (max_age_days - min_age_days)
        df['age_normalized'] = df['age_normalized'].clip(0, 1)  # Ensure values are in [0,1]
        
        # Age risk exponential with safety check
        df['age_risk_exponential'] = np.where(age_years > 45, 
                                            np.exp(np.clip((age_years - 45) / 10, 0, 5)), 
                                            1.0)
        
        # Blood pressure features
        try:
            bp_bins = [0, 120, 140, 160, 180, 1000]
            bp_labels = ['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis']
            df['bp_category'] = pd.cut([df['ap_hi'].iloc[0]], bins=bp_bins, labels=bp_labels, include_lowest=True)[0]
        except Exception as e:
            logger.warning(f"Error creating BP categories: {e}")
            df['bp_category'] = 'Normal'  # Default value
            
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

        # Metabolic features with safety checks
        df['metabolic_profile'] = df['cholesterol'] / max(age_years, 1)  # Avoid division by zero
        
        try:
            chol_bins = [0, 1.5, 2.5, 3.5, 1000]
            chol_labels = ['Normal', 'Above Normal', 'Well Above Normal', 'High']
            df['chol_category'] = pd.cut([df['cholesterol'].iloc[0]], bins=chol_bins, labels=chol_labels, include_lowest=True)[0]
        except Exception as e:
            logger.warning(f"Error creating cholesterol categories: {e}")
            df['chol_category'] = 'Normal'  # Default value
        
        df['metabolic_syndrome_risk'] = (
            (df['cholesterol'] > 1).astype(int) + 
            (df['gluc'] > 1).astype(int) + 
            (df['ap_hi'] > 140).astype(int)
        )
        
        # Gender interaction features
        df['male_age_interaction'] = (df['gender'] == 2).astype(int) * age_years
        df['female_chol_interaction'] = (df['gender'] == 1).astype(int) * df['cholesterol']
        df['gender_specific_risk'] = np.where(
            df['gender'] == 1, 
            df['cholesterol'] * 0.008, 
            age_years * 0.1 + df['cholesterol'] * 0.005
        )

        # Composite risk scores
        df['traditional_risk_score'] = (
            age_years * 0.04 + 
            df['gender'] * 10 + 
            (df['cholesterol'] - 1) * 20 + 
            df['ap_hi'] * 0.1 + 
            df['gluc'] * 20
        )
        df['cardiac_risk_score'] = (df['pulse_pressure'] * 0.2 + df['ap_hi'] * 0.1)
        df['combined_risk_score'] = (df['traditional_risk_score'] * 0.4 + df['cardiac_risk_score'] * 0.6)
        
        # Categorical encoding with safer approach
        # Age group encoding
        if age_years < 45:
            df['age_group_encoded'] = 0
        elif age_years < 55:
            df['age_group_encoded'] = 1
        elif age_years < 65:
            df['age_group_encoded'] = 2
        else:
            df['age_group_encoded'] = 3
            
        # Cholesterol category encoding
        chol_val = df['cholesterol'].iloc[0]
        if chol_val <= 1.5:
            df['chol_category_encoded'] = 0
        elif chol_val <= 2.5:
            df['chol_category_encoded'] = 1
        elif chol_val <= 3.5:
            df['chol_category_encoded'] = 2
        else:
            df['chol_category_encoded'] = 3
            
        # Blood pressure category encoding
        bp_val = df['ap_hi'].iloc[0]
        if bp_val < 120:
            df['bp_category_encoded'] = 0
        elif bp_val < 140:
            df['bp_category_encoded'] = 1
        elif bp_val < 160:
            df['bp_category_encoded'] = 2
        elif bp_val < 180:
            df['bp_category_encoded'] = 3
        else:
            df['bp_category_encoded'] = 4

        logger.info(f"V3 Feature engineering completed successfully. DataFrame shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error in V3 feature engineering: {e}")
        st.error(f"Error processing your data for V3: {str(e)}")
        return None
    
def create_feature_engineering_v4_with_scaler_features(patient_data, model_data):
    """
    Generate features using the EXACT feature names from the scaler (30 features).
    This is the correct approach based on the logs.
    """
    try:
        # Get the exact feature names from the scaler (these are the correct ones)
        scaler = model_data.get('scaler')
        if scaler and hasattr(scaler, 'feature_names_in_'):
            feature_names = list(scaler.feature_names_in_)
            logger.info(f"Using SCALER feature names: {len(feature_names)} features")
            logger.info(f"Scaler features: {feature_names}")
        else:
            logger.error("Could not get feature names from scaler")
            return create_feature_engineering_v4_fixed(patient_data)
        
        age_years = float(patient_data['age'])
        features = {}
        
        # Generate each feature based on the exact scaler feature names
        for feat_name in feature_names:
            if feat_name == 'age':
                features[feat_name] = float(age_years * 365.25)  # Convert to days
                
            elif feat_name in ['ap_hi', 'ap_lo', 'cholesterol', 'weight', 'height']:
                # Direct patient data features
                features[feat_name] = float(patient_data[feat_name])
                
            elif feat_name == 'age_normalized':
                min_age_days = 25 * 365.25
                max_age_days = 70 * 365.25
                age_days = age_years * 365.25
                age_norm = (age_days - min_age_days) / (max_age_days - min_age_days)
                features[feat_name] = float(max(0.0, min(1.0, age_norm)))
                
            elif feat_name == 'age_risk_exponential':
                if age_years > 45:
                    exp_input = min(5.0, (age_years - 45) / 10)
                    features[feat_name] = float(np.exp(exp_input))
                else:
                    features[feat_name] = 1.0
                    
            elif feat_name == 'age_squared':
                features[feat_name] = float(age_years ** 2)
                
            elif feat_name == 'age_log':
                features[feat_name] = float(np.log1p(age_years))
                
            elif feat_name == 'age_risk_category':
                if age_years <= 45:
                    features[feat_name] = 0.0
                elif age_years <= 55:
                    features[feat_name] = 1.0
                elif age_years <= 65:
                    features[feat_name] = 2.0
                else:
                    features[feat_name] = 3.0
                    
            elif feat_name == 'traditional_risk_score':
                features[feat_name] = float(
                    age_years * 0.04 + 
                    patient_data['gender'] * 10 + 
                    (patient_data['cholesterol'] - 1) * 20 + 
                    patient_data['ap_hi'] * 0.1 + 
                    patient_data['gluc'] * 20
                )
                
            elif feat_name == 'metabolic_profile':
                features[feat_name] = float(patient_data['cholesterol'] / max(age_years, 1))
                
            elif feat_name == 'male_age_interaction':
                is_male = 1.0 if patient_data['gender'] == 2 else 0.0
                features[feat_name] = float(is_male * age_years)
                
            elif feat_name == 'combined_risk_score':
                trad_risk = (
                    age_years * 0.04 + 
                    patient_data['gender'] * 10 + 
                    (patient_data['cholesterol'] - 1) * 20 + 
                    patient_data['ap_hi'] * 0.1 + 
                    patient_data['gluc'] * 20
                )
                pulse_pressure = patient_data['ap_hi'] - patient_data['ap_lo']
                cardiac_risk = pulse_pressure * 0.2 + patient_data['ap_hi'] * 0.1
                features[feat_name] = float(trad_risk * 0.4 + cardiac_risk * 0.6)
                
            elif feat_name == 'framingham_score':
                features[feat_name] = float(
                    age_years * 0.04 + 
                    (patient_data['ap_hi'] - 120) * 0.02 + 
                    patient_data['cholesterol'] * 15
                )
                
            elif feat_name == 'mean_arterial_pressure':
                pulse_pressure = patient_data['ap_hi'] - patient_data['ap_lo']
                features[feat_name] = float(patient_data['ap_lo'] + (pulse_pressure / 3))
                
            # Polynomial features - handle spaces in feature names
            elif feat_name == 'poly_age ap_hi':
                features[feat_name] = float(age_years * patient_data['ap_hi'])
                
            elif feat_name == 'poly_age ap_lo':
                features[feat_name] = float(age_years * patient_data['ap_lo'])
                
            elif feat_name == 'poly_age cholesterol':
                features[feat_name] = float(age_years * patient_data['cholesterol'])
                
            elif feat_name == 'poly_age gluc':
                features[feat_name] = float(age_years * patient_data['gluc'])
                
            elif feat_name == 'poly_ap_hi ap_lo':
                features[feat_name] = float(patient_data['ap_hi'] * patient_data['ap_lo'])
                
            elif feat_name == 'poly_ap_hi cholesterol':
                features[feat_name] = float(patient_data['ap_hi'] * patient_data['cholesterol'])
                
            elif feat_name == 'poly_ap_hi gluc':
                features[feat_name] = float(patient_data['ap_hi'] * patient_data['gluc'])
                
            elif feat_name == 'poly_ap_lo cholesterol':
                features[feat_name] = float(patient_data['ap_lo'] * patient_data['cholesterol'])
                
            # Statistical aggregation features
            elif feat_name in ['feature_mean', 'feature_std', 'feature_median', 'feature_max', 'feature_range']:
                key_features = [age_years, patient_data['ap_hi'], patient_data['ap_lo'], 
                               patient_data['cholesterol'], patient_data['gluc']]
                
                if feat_name == 'feature_mean':
                    features[feat_name] = float(np.mean(key_features))
                elif feat_name == 'feature_std':
                    features[feat_name] = float(np.std(key_features))
                elif feat_name == 'feature_median':
                    features[feat_name] = float(np.median(key_features))
                elif feat_name == 'feature_max':
                    features[feat_name] = float(np.max(key_features))
                elif feat_name == 'feature_range':
                    features[feat_name] = float(np.max(key_features) - np.min(key_features))
                    
            else:
                # For any unrecognized feature, use default value
                features[feat_name] = 0.0
                logger.warning(f"Unknown feature '{feat_name}', using default value 0.0")
        
        # Create DataFrame with exact feature order from scaler
        final_df = pd.DataFrame([features])
        
        # Reorder columns to match scaler exactly
        final_df = final_df.reindex(columns=feature_names, fill_value=0.0)
        
        logger.info(f"V4 SCALER-BASED: Generated exactly {len(final_df.columns)} features")
        logger.info(f"V4 SCALER-BASED features: {list(final_df.columns)}")
        
        # Verify we have exactly 30 features
        if len(final_df.columns) != 30:
            logger.error(f"MISMATCH: Expected 30 features, got {len(final_df.columns)}")
            return None
            
        # Verify all features are present
        missing_features = [feat for feat in feature_names if feat not in final_df.columns]
        if missing_features:
            logger.error(f"MISSING FEATURES: {missing_features}")
            return None
            
        return final_df
        
    except Exception as e:
        logger.error(f"Error in V4 scaler-based feature engineering: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_feature_engineering_v4_fixed(patient_data):
    """Generate exactly 30 features that V4 model expects (CORRECTED VERSION)."""
    try:
        age_years = float(patient_data['age'])
        
        logger.info(f"PROCESSING V4 FIXED: age={age_years}")
        
        # V4 needs 30 features, not 20 as originally implemented
        features = {}
        
        logger.info("CREATING ALL 30 FEATURES FOR V4...")
        
        # 1. Original features (11 base features)
        features['age'] = float(age_years * 365.25)  # Convert to days
        features['cholesterol'] = float(patient_data['cholesterol'])
        features['weight'] = float(patient_data['weight'])
        features['gluc'] = float(patient_data['gluc'])
        features['ap_lo'] = float(patient_data['ap_lo'])
        features['ap_hi'] = float(patient_data['ap_hi'])
        features['active'] = float(patient_data['active'])
        features['smoke'] = float(patient_data['smoke'])
        features['height'] = float(patient_data['height'])
        features['gender'] = float(patient_data['gender'])
        features['alco'] = float(patient_data['alco'])
        
        # 2. Age-derived features
        min_age_days = 25 * 365.25
        max_age_days = 70 * 365.25
        age_norm = (features['age'] - min_age_days) / (max_age_days - min_age_days)
        features['age_normalized'] = float(max(0.0, min(1.0, age_norm)))
        
        # Age risk exponential
        if age_years > 45:
            exp_input = min(5.0, (age_years - 45) / 10)
            age_risk_exp = float(2.71828 ** exp_input)
        else:
            age_risk_exp = 1.0
        features['age_risk_exponential'] = float(age_risk_exp)
        
        # Additional age features
        features['age_squared'] = float(age_years ** 2)
        features['age_log'] = float(np.log1p(age_years))
        
        # Age risk category
        if age_years <= 45:
            features['age_risk_category'] = 0.0
        elif age_years <= 55:
            features['age_risk_category'] = 1.0
        elif age_years <= 65:
            features['age_risk_category'] = 2.0
        else:
            features['age_risk_category'] = 3.0
        
        # 3. Blood pressure features
        features['pulse_pressure'] = float(patient_data['ap_hi'] - patient_data['ap_lo'])
        features['mean_arterial_pressure'] = float(patient_data['ap_lo'] + (features['pulse_pressure'] / 3))
        
        # 4. Polynomial interaction features (based on scaler error)
        features['poly_age_cholesterol'] = float(age_years * patient_data['cholesterol'])
        features['poly_age_ap_hi'] = float(age_years * patient_data['ap_hi'])
        features['poly_age_ap_lo'] = float(age_years * patient_data['ap_lo'])
        features['poly_age_gluc'] = float(age_years * patient_data['gluc'])
        features['poly_ap_hi_ap_lo'] = float(patient_data['ap_hi'] * patient_data['ap_lo'])
        features['poly_ap_hi_cholesterol'] = float(patient_data['ap_hi'] * patient_data['cholesterol'])
        features['poly_ap_hi_gluc'] = float(patient_data['ap_hi'] * patient_data['gluc'])
        features['poly_ap_lo_cholesterol'] = float(patient_data['ap_lo'] * patient_data['cholesterol'])
        
        # 5. Statistical aggregation features
        key_features = [age_years, patient_data['ap_hi'], patient_data['ap_lo'], 
                       patient_data['cholesterol'], patient_data['gluc']]
        
        features['feature_mean'] = float(np.mean(key_features))
        features['feature_std'] = float(np.std(key_features))
        features['feature_median'] = float(np.median(key_features))
        features['feature_max'] = float(np.max(key_features))
        features['feature_range'] = float(features['feature_max'] - np.min(key_features))
        
        # 6. Medical risk scores
        features['framingham_score'] = float(
            age_years * 0.04 + 
            (patient_data['ap_hi'] - 120) * 0.02 + 
            patient_data['cholesterol'] * 15
        )
        
        # 7. Gender-specific features
        is_male = 1.0 if patient_data['gender'] == 2 else 0.0
        is_female = 1.0 if patient_data['gender'] == 1 else 0.0
        
        features['male_age_interaction'] = float(is_male * age_years)
        features['female_chol_interaction'] = float(is_female * patient_data['cholesterol'])
        
        # Gender specific risk
        if patient_data['gender'] == 1:
            gender_risk = patient_data['cholesterol'] * 0.008
        else:
            gender_risk = age_years * 0.1 + patient_data['cholesterol'] * 0.005
        features['gender_specific_risk'] = float(gender_risk)
        
        # 8. Composite scores
        features['traditional_risk_score'] = float(
            age_years * 0.04 + 
            patient_data['gender'] * 10 + 
            (patient_data['cholesterol'] - 1) * 20 + 
            patient_data['ap_hi'] * 0.1 + 
            patient_data['gluc'] * 20
        )
        
        features['metabolic_profile'] = float(patient_data['cholesterol'] / max(age_years, 1))
        
        cardiac_risk = features['pulse_pressure'] * 0.2 + patient_data['ap_hi'] * 0.1
        features['combined_risk_score'] = float(features['traditional_risk_score'] * 0.4 + cardiac_risk * 0.6)
        
        # Verify we have exactly 30 features
        logger.info(f"V4 FIXED: Generated {len(features)} features")
        
        if len(features) != 30:
            # If we don't have 30, add additional features to complete
            missing_count = 30 - len(features)
            logger.warning(f"Need to add {missing_count} more features to reach 30")
            
            # Add additional features based on what's missing in the scaler
            additional_features = [
                'bmi_category', 'lifestyle_score', 'bp_risk_category', 'metabolic_syndrome_risk',
                'age_chol_risk', 'combined_lifestyle_risk', 'cardiovascular_age', 'risk_multiplier'
            ]
            
            for i, feat_name in enumerate(additional_features[:missing_count]):
                if feat_name == 'bmi_category':
                    height_m = patient_data['height'] / 100
                    bmi = patient_data['weight'] / (height_m ** 2)
                    if bmi < 18.5:
                        features[feat_name] = 0.0
                    elif bmi < 25:
                        features[feat_name] = 1.0
                    elif bmi < 30:
                        features[feat_name] = 2.0
                    else:
                        features[feat_name] = 3.0
                elif feat_name == 'lifestyle_score':
                    features[feat_name] = float(patient_data['smoke'] + patient_data['alco'] + (1 - patient_data['active']))
                elif feat_name == 'bp_risk_category':
                    if patient_data['ap_hi'] < 120:
                        features[feat_name] = 0.0
                    elif patient_data['ap_hi'] < 140:
                        features[feat_name] = 1.0
                    elif patient_data['ap_hi'] < 160:
                        features[feat_name] = 2.0
                    else:
                        features[feat_name] = 3.0
                elif feat_name == 'metabolic_syndrome_risk':
                    features[feat_name] = float(
                        (patient_data['cholesterol'] > 1) + 
                        (patient_data['gluc'] > 1) + 
                        (patient_data['ap_hi'] > 140)
                    )
                else:
                    # Generic calculated features
                    features[feat_name] = float(np.random.uniform(0, 1))  # Temporary until we have exact names
        
        # Ensure we have exactly 30 features
        feature_names = list(features.keys())
        if len(feature_names) > 30:
            feature_names = feature_names[:30]
        elif len(feature_names) < 30:
            # Add dummy features if necessary
            for i in range(len(feature_names), 30):
                features[f'dummy_feature_{i}'] = 0.0
                
        # Create DataFrame with exactly 30 features
        final_features = {name: features[name] for name in list(features.keys())[:30]}
        final_df = pd.DataFrame([final_features])
        
        logger.info(f"V4 FIXED: Final feature count: {len(final_df.columns)}")
        logger.info(f"V4 FIXED features: {list(final_df.columns)}")
        
        return final_df
        
    except Exception as e:
        logger.error(f"Error in V4 FIXED feature engineering: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_feature_engineering_v4_with_model_inspection(patient_data, model_data):
    """
    Updated version that uses scaler features instead of model feature_names.
    """
    try:
        # Use scaler features since they are the correct ones (30 features)
        return create_feature_engineering_v4_with_scaler_features(patient_data, model_data)
            
    except Exception as e:
        logger.error(f"Error in model inspection approach: {e}")
        return create_feature_engineering_v4_fixed(patient_data)

def create_fallback_prediction_v4(patient_data):
    """Fallback prediction method if feature engineering fails."""
    try:
        # Simple rule-based prediction as fallback
        age_years = patient_data['age']
        
        # Calculate basic risk factors
        age_risk = min(age_years / 80, 1.0)  # Age normalized to 0-1
        bp_risk = (patient_data['ap_hi'] - 90) / (200 - 90)  # BP normalized
        chol_risk = (patient_data['cholesterol'] - 1) / 2  # Cholesterol normalized
        glucose_risk = (patient_data['gluc'] - 1) / 2  # Glucose normalized
        lifestyle_risk = (patient_data['smoke'] + patient_data['alco'] + (1 - patient_data['active'])) / 3
        
        # Simple weighted average
        probability = (
            age_risk * 0.3 +
            bp_risk * 0.25 +
            chol_risk * 0.2 +
            glucose_risk * 0.15 +
            lifestyle_risk * 0.1
        )
        
        probability = np.clip(probability, 0, 1)
        
        # Determine risk category
        if probability < 0.3:
            risk_category, risk_class = "Low", "risk-low"
        elif probability < 0.6:
            risk_category, risk_class = "Moderate", "risk-moderate"
        elif probability < 0.8:
            risk_category, risk_class = "High", "risk-high"
        else:
            risk_category, risk_class = "Critical", "risk-critical"
        
        return {
            'probability': probability,
            'risk_category': risk_category,
            'risk_class': risk_class,
            'model_used': 'Heart Risk AI V4 (Simplified)'
        }
        
    except Exception as e:
        logger.error(f"Fallback prediction error: {e}")
        return None

def create_feature_engineering(patient_data, model_key, model_data=None):
    """Apply the appropriate feature engineering based on the model."""
    if model_key == "heart_risk_ensemble_v3":
        return create_feature_engineering_v3(patient_data)
    elif model_key == "heart_risk_ai_v4":
        # ALWAYS use scaler-based approach for V4 since scaler has the correct 30 features
        if model_data is not None:
            return create_feature_engineering_v4_with_scaler_features(patient_data, model_data)
        else:
            logger.error("V4 requires model_data to access scaler features")
            return create_feature_engineering_v4_fixed(patient_data)
    else:
        logger.error(f"Unknown model key: {model_key}")
        st.error(f"Unknown model: {model_key}")
        return None

# --- Prediction ---
def make_prediction(model_data, patient_data_engineered, model_info, patient_data_original=None):
    """Make a prediction with the model."""
    if model_data is None or patient_data_engineered is None:
        return None
        
    try:
        # Debug: Show what features the model expects vs what we have
        debug_model_features(model_data, model_info['name'])

        # --- CORRECCIÓN: Para V4, usar siempre los features del scaler ---
        if "v4" in model_info['name'].lower() and 'scaler' in model_data and hasattr(model_data['scaler'], 'feature_names_in_'):
            feature_names = list(model_data['scaler'].feature_names_in_)
            logger.info(f"V4: Usando features del scaler ({len(feature_names)})")
        else:
            feature_names = model_data.get('feature_names', [])
            if not feature_names:
                logger.warning("No feature names found in model data. Using available columns.")
                feature_names = patient_data_engineered.columns.tolist()
        
        logger.info(f"Patient data has {len(patient_data_engineered.columns)} features: {patient_data_engineered.columns.tolist()}")
        logger.info(f"Model expects {len(feature_names)} features: {feature_names}")
        
        # Check exact feature matching
        patient_features = set(patient_data_engineered.columns)
        expected_features = set(feature_names)
        missing_features = expected_features - patient_features
        extra_features = patient_features - expected_features
        
        if missing_features:
            logger.error(f"MISSING FEATURES: {missing_features}")
        if extra_features:
            logger.warning(f"EXTRA FEATURES: {extra_features}")
        
        # Prepare features for prediction - exact order and features
        X = patient_data_engineered.reindex(columns=feature_names, fill_value=0)
        logger.info(f"Feature matrix shape after reindexing: {X.shape}")
        logger.info(f"Feature matrix columns: {X.columns.tolist()}")
        
        # Check for any non-numeric data
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            logger.error(f"NON-NUMERIC COLUMNS FOUND: {non_numeric_cols.tolist()}")
            for col in non_numeric_cols:
                try:
                    unique_vals = X[col].unique()
                    logger.error(f"Column '{col}' has values: {unique_vals}")
                except Exception as unique_error:
                    logger.error(f"Column '{col}' has values that can't be displayed: {unique_error}")
                    logger.error(f"Column '{col}' dtype: {X[col].dtype}")
                    logger.error(f"Column '{col}' sample values: {X[col].iloc[:3].tolist()}")
        
        # Ensure we have numeric data
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        logger.info(f"Numeric data shape: {X_numeric.shape}")
        
        if X_numeric.shape[1] != len(feature_names):
            raise ValueError(f"Feature count mismatch: expected {len(feature_names)}, got {X_numeric.shape[1]}")
        
        # Check for infinite or NaN values
        if np.any(np.isinf(X_numeric.values)):
            logger.error("INFINITE VALUES FOUND in feature matrix")
            inf_cols = X_numeric.columns[np.isinf(X_numeric).any()].tolist()
            logger.error(f"Columns with infinite values: {inf_cols}")
            X_numeric = X_numeric.replace([np.inf, -np.inf], 0)
        
        if np.any(np.isnan(X_numeric.values)):
            logger.error("NaN VALUES FOUND in feature matrix")
            nan_cols = X_numeric.columns[np.isnan(X_numeric).any()].tolist()
            logger.error(f"Columns with NaN values: {nan_cols}")
            X_numeric = X_numeric.fillna(0)
        
        # Scale features if scaler is available
        if 'scaler' in model_data and model_data['scaler'] is not None:
            logger.info("Applying scaler to features")
            try:
                # Para V4, solo escalar si los features coinciden exactamente
                if "v4" in model_info['name'].lower():
                    scaler_features = getattr(model_data['scaler'], 'feature_names_in_', None)
                    current_features = set(X_numeric.columns)
                    if scaler_features is not None:
                        expected_features = set(scaler_features)
                        missing = expected_features - current_features
                        extra = current_features - expected_features
                        if missing or extra:
                            logger.warning(f"V4 Scaler mismatch detected. Missing: {missing}, Extra: {extra}")
                            logger.warning("Using unscaled features for V4 model...")
                            X_scaled = X_numeric.values
                        else:
                            X_scaled = model_data['scaler'].transform(X_numeric)
                    else:
                        X_scaled = model_data['scaler'].transform(X_numeric)
                else:
                    X_scaled = model_data['scaler'].transform(X_numeric)
                logger.info(f"Scaled data shape: {X_scaled.shape}")
            except Exception as scaler_error:
                logger.error(f"SCALER ERROR: {scaler_error}")
                logger.warning("Falling back to unscaled features...")
                X_scaled = X_numeric.values
        else:
            logger.warning("No scaler found in model data. Using unscaled features.")
            X_scaled = X_numeric.values
        
        # Get the model
        model = model_data.get('ensemble_model') or model_data.get('model')
        if model is None:
            raise ValueError("No model found in model data")
        
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")
        
        # Make prediction
        try:
            if hasattr(model, 'predict_proba'):
                logger.info("Using predict_proba method")
                proba_result = model.predict_proba(X_scaled)
                logger.info(f"Predict_proba result shape: {proba_result.shape}")
                logger.info(f"Predict_proba result: {proba_result}")
                probability = proba_result[0, 1]
            else:
                if hasattr(model, 'decision_function'):
                    logger.info("Using decision_function method")
                    score = model.decision_function(X_scaled)[0]
                    probability = 1 / (1 + np.exp(-score))
                else:
                    logger.info("Using predict method")
                    probability = model.predict(X_scaled)[0]
        except Exception as prediction_error:
            logger.error(f"PREDICTION ERROR: {prediction_error}")
            raise prediction_error
        
        probability = np.clip(probability, 0, 1)
        logger.info(f"Final probability: {probability}")
        
        if probability < 0.3:
            risk_category, risk_class = "Low", "risk-low"
        elif probability < 0.6:
            risk_category, risk_class = "Moderate", "risk-moderate"
        elif probability < 0.8:
            risk_category, risk_class = "High", "risk-high"
        else:
            risk_category, risk_class = "Critical", "risk-critical"
        
        logger.info(f"Prediction successful with {model_info['name']}: {probability:.3f} ({risk_category})")
        return {
            'probability': probability, 
            'risk_category': risk_category, 
            'risk_class': risk_class,
            'model_used': model_info['name']
        }
        
    except Exception as e:
        logger.error(f"DETAILED PREDICTION ERROR for {model_info['name']}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # For model V4, try fallback prediction
        if "v4" in model_info['name'].lower() and patient_data_original:
            logger.info("Attempting fallback prediction for V4...")
            st.warning("⚠️ Using simplified prediction method for V4 due to feature compatibility issues.")
            st.error(f"**Debug Error**: {str(e)}")
            
            fallback_result = create_fallback_prediction_v4(patient_data_original)
            if fallback_result:
                return fallback_result
        
        st.error(f"Error making prediction with {model_info['name']}: {str(e)}")
        
        # Show additional debugging info in the UI
        with st.expander("🔧 Debug Information"):
            st.write(f"**Error Type**: {type(e).__name__}")
            st.write(f"**Error Message**: {str(e)}")
            
            st.write("**Model expects these features:**")
            st.write(f"Total: {len(feature_names)} features")
            st.code(", ".join(feature_names))
            
            st.write("**Generated features:**")
            if patient_data_engineered is not None:
                st.write(f"Total: {len(patient_data_engineered.columns)} features")
                st.code(", ".join(patient_data_engineered.columns.tolist()))
                patient_features = set(patient_data_engineered.columns)
                expected_features = set(feature_names)
                missing = expected_features - patient_features
                extra = patient_features - expected_features
                if missing:
                    st.write(f"**Missing features**: {missing}")
                if extra:
                    st.write(f"**Extra features**: {extra}")
                if not missing and not extra:
                    st.write("✅ **Feature names match perfectly**")
        
        return None

# --- UI Components ---
def create_model_selector():
    """Create model selection interface."""
    st.markdown('<div class="sub-header">🤖 Select AI Model</div>', unsafe_allow_html=True)
    
    model_options = [f"{info['name']} - {info['type']}" for info in AVAILABLE_MODELS.values()]
    model_keys = list(AVAILABLE_MODELS.keys())
    
    selected_idx = st.selectbox(
        "Choose the AI model for your risk assessment:",
        range(len(model_options)),
        format_func=lambda x: model_options[x],
        index=1  # Default to V4 model
    )
    
    selected_model_key = model_keys[selected_idx]
    selected_model_info = AVAILABLE_MODELS[selected_model_key]
    
    # Display model information
    st.markdown(f"""
    <div class="model-info">
        <strong>🔬 {selected_model_info['name']}</strong><br>
        <strong>Type:</strong> {selected_model_info['type']}<br>
        <strong>Description:</strong> {selected_model_info['description']}
    </div>
    """, unsafe_allow_html=True)
    
    return selected_model_key, selected_model_info

def create_input_form():
    """Create the input form for user data."""
    st.markdown('<div class="sub-header">📋 Enter Your Medical Data</div>', unsafe_allow_html=True)
    patient_data = {}
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Personal Information")
        patient_data['age'] = st.slider("Age", 18, 80, 50, help="Your age in years.")
        gender_display = st.selectbox("Gender", ["Female", "Male"], index=0)
        patient_data['height'] = st.slider("Height (cm)", 140, 220, 170)
        patient_data['weight'] = st.slider("Weight (kg)", 40, 150, 70)
        
        # Map gender to numeric values (keeping original mapping)
        patient_data['gender'] = {"Female": 1, "Male": 2}[gender_display]

    with col2:
        st.subheader("🩺 Clinical Measurements")
        patient_data['ap_hi'] = st.slider("Systolic blood pressure", 90, 250, 120, 
                                        help="The upper number in a blood pressure reading (mmHg).")
        patient_data['ap_lo'] = st.slider("Diastolic blood pressure", 60, 150, 80, 
                                        help="The lower number in a blood pressure reading (mmHg).")
        
        cholesterol_display = st.selectbox("Cholesterol Level", 
                                         ["Normal", "Above Normal", "Well Above Normal"], 
                                         index=0)
        glucose_display = st.selectbox("Glucose Level", 
                                     ["Normal", "Above Normal", "Well Above Normal"], 
                                     index=0)

        # Map clinical values (keeping original mapping)
        cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        glucose_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        patient_data['cholesterol'] = cholesterol_map[cholesterol_display]
        patient_data['gluc'] = glucose_map[glucose_display]

    with st.expander("🏃 Lifestyle Factors"):
        smoke_display = st.radio("Do you smoke?", ["No", "Yes"], index=0)
        alcohol_display = st.radio("Do you consume alcohol?", ["No", "Yes"], index=0)
        active_display = st.radio("Are you physically active?", ["No", "Yes"], index=1)
        
        # Map lifestyle values (keeping original mapping)
        patient_data['smoke'] = {"No": 0, "Yes": 1}[smoke_display]
        patient_data['alco'] = {"No": 0, "Yes": 1}[alcohol_display]
        patient_data['active'] = {"No": 0, "Yes": 1}[active_display]
        
    return patient_data

def display_results(prediction_result):
    """Display prediction results including gauge and risk category."""
    st.markdown('<div class="sub-header">📊 Your Risk Assessment</div>', unsafe_allow_html=True)
    
    prob = prediction_result['probability']
    risk_cat = prediction_result['risk_category']
    risk_cls = prediction_result['risk_class']
    model_used = prediction_result['model_used']
    
    st.markdown(f'''
    <div class="risk-box {risk_cls}">
        <h2>{risk_cat} Risk</h2>
        <h3>{prob:.1%} probability of cardiovascular disease</h3>
        <p><em>Analysis performed by: {model_used}</em></p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cardiovascular Risk Meter (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellowgreen"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_risk_breakdown_and_recommendations(patient_data, engineered_df):
    """Analyzes risk factors and provides tailored recommendations."""
    st.markdown('<div class="sub-header">🔍 Risk Factor Analysis & Recommendations</div>', unsafe_allow_html=True)
    
    # --- Risk Factor Analysis ---
    st.subheader("Key Contributing Factors")
    positive_factors = []
    negative_factors = []

    # Analyze factors with improved logic
    # Blood Pressure Analysis
    if patient_data['ap_hi'] >= 140 or patient_data['ap_lo'] >= 90:
        severity = "High" if patient_data['ap_hi'] >= 160 or patient_data['ap_lo'] >= 100 else "Elevated"
        negative_factors.append(f"**{severity} Blood Pressure:** Your reading ({patient_data['ap_hi']}/{patient_data['ap_lo']} mmHg) indicates hypertension.")
    elif patient_data['ap_hi'] >= 120:
        negative_factors.append(f"**Elevated Blood Pressure:** Your reading ({patient_data['ap_hi']}/{patient_data['ap_lo']} mmHg) is above optimal.")
    else:
        positive_factors.append(f"**Optimal Blood Pressure:** Your reading ({patient_data['ap_hi']}/{patient_data['ap_lo']} mmHg) is excellent.")

    # Cholesterol Analysis
    if patient_data['cholesterol'] > 2:
        negative_factors.append("**High Cholesterol:** Well above normal levels significantly increase cardiovascular risk.")
    elif patient_data['cholesterol'] > 1:
        negative_factors.append("**Elevated Cholesterol:** Above normal levels may contribute to plaque buildup.")
    else:
        positive_factors.append("**Normal Cholesterol:** Healthy cholesterol levels protect your heart.")

    # Glucose Analysis
    if patient_data['gluc'] > 2:
        negative_factors.append("**High Glucose:** Well above normal levels indicate diabetes risk.")
    elif patient_data['gluc'] > 1:
        negative_factors.append("**Elevated Glucose:** Above normal levels may indicate pre-diabetes.")
    else:
        positive_factors.append("**Normal Glucose:** Healthy blood sugar levels.")

    # Age Analysis
    if patient_data['age'] >= 65:
        negative_factors.append(f"**Advanced Age:** At {patient_data['age']} years, age is a significant non-modifiable risk factor.")
    elif patient_data['age'] >= 55:
        negative_factors.append(f"**Age Factor:** At {patient_data['age']} years, cardiovascular risk begins to increase.")

    # Lifestyle Factors
    if patient_data['smoke'] == 1:
        negative_factors.append("**Smoking:** Smoking dramatically increases cardiovascular disease risk.")
    else:
        positive_factors.append("**Non-Smoker:** Not smoking significantly protects your cardiovascular health.")
        
    if patient_data['active'] == 0:
        negative_factors.append("**Physical Inactivity:** Sedentary lifestyle increases cardiovascular risk.")
    else:
        positive_factors.append("**Physically Active:** Regular physical activity strengthens your heart and circulation.")

    if patient_data['alco'] == 1:
        negative_factors.append("**Alcohol Consumption:** Regular alcohol use may contribute to cardiovascular risk.")

    # Display factors
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ❤️ Protective Factors")
        if positive_factors:
            for factor in positive_factors:
                st.success(factor)
        else:
            st.info("Consider adopting heart-healthy lifestyle changes.")
            
    with col2:
        st.markdown("#### ⚠️ Risk Factors")
        if negative_factors:
            for factor in negative_factors:
                st.warning(factor)
        else:
            st.success("No major risk factors identified. Excellent!")

    # --- Personalized Recommendations ---
    st.subheader("Personalized Recommendations")
    st.markdown("""
    <div class="recommendation-box">
    <strong>Important:</strong> These are general recommendations based on your profile. 
    Always consult with a qualified healthcare professional for personalized medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    recommendations = []
    
    # Blood pressure recommendations
    if patient_data['ap_hi'] >= 140 or patient_data['ap_lo'] >= 90:
        recommendations.append("🩺 **Blood Pressure Management:** Consult your doctor about blood pressure control strategies, including medication if needed.")
        recommendations.append("🧂 **Reduce Sodium:** Limit salt intake to less than 2,300mg per day (ideally 1,500mg).")
    
    # Cholesterol recommendations
    if patient_data['cholesterol'] > 1:
        recommendations.append("🥗 **Heart-Healthy Diet:** Focus on fruits, vegetables, whole grains, and lean proteins. Limit saturated fats.")
        recommendations.append("🏃 **Increase Physical Activity:** Aim for at least 150 minutes of moderate aerobic activity per week.")
    
    # Smoking recommendations
    if patient_data['smoke'] == 1:
        recommendations.append("🚭 **Quit Smoking:** This is the single most important change you can make. Seek professional support.")
    
    # Activity recommendations
    if patient_data['active'] == 0:
        recommendations.append("💪 **Start Moving:** Begin with 30 minutes of moderate activity most days. Even walking counts!")
    
    # Age-related recommendations
    if patient_data['age'] >= 55:
        recommendations.append("📅 **Regular Check-ups:** Schedule annual cardiovascular screenings with your healthcare provider.")
    
    # General recommendations
    recommendations.extend([
        "😴 **Quality Sleep:** Aim for 7-9 hours of quality sleep each night.",
        "🧘 **Stress Management:** Practice stress-reduction techniques like meditation or deep breathing.",
        "⚖️ **Maintain Healthy Weight:** If overweight, even modest weight loss can reduce cardiovascular risk."
    ])
    
    # Display recommendations
    for i, rec in enumerate(recommendations[:6], 1):  # Show top 6 recommendations
        st.info(f"{i}. {rec}")
    
    # Success message for low-risk individuals
    if len(negative_factors) <= 1:
        st.balloons()
        st.success("🎉 Your profile shows good cardiovascular health! Keep up the excellent work with regular exercise, healthy eating, and routine check-ups.")

def display_population_comparison(patient_data, stats_df):
    """Compares user's data with the dataset's population."""
    if stats_df is None:
        return
        
    st.markdown('<div class="sub-header">📈 Population Comparison</div>', unsafe_allow_html=True)
    st.write("Here's how your data compares to the population average:")
    
    try:
        avg_age = stats_df['age_years'].mean()
        avg_ap_hi = stats_df['ap_hi'].mean()
        avg_ap_lo = stats_df['ap_lo'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            age_diff = patient_data['age'] - avg_age
            st.metric("Your Age", f"{patient_data['age']:.0f} years", 
                     f"{age_diff:+.1f} vs avg ({avg_age:.1f})")
        with col2:
            bp_hi_diff = patient_data['ap_hi'] - avg_ap_hi
            st.metric("Your Systolic BP", f"{patient_data['ap_hi']} mmHg", 
                     f"{bp_hi_diff:+.1f} vs avg ({avg_ap_hi:.1f})")
        with col3:
            bp_lo_diff = patient_data['ap_lo'] - avg_ap_lo
            st.metric("Your Diastolic BP", f"{patient_data['ap_lo']} mmHg", 
                     f"{bp_lo_diff:+.1f} vs avg ({avg_ap_lo:.1f})")

        # Age distribution chart
        fig = px.histogram(stats_df, x='age_years', nbins=30, 
                          title="Your Age vs. Population Distribution",
                          labels={'age_years': 'Age (years)', 'count': 'Number of People'})
        fig.add_vline(x=patient_data['age'], line_width=3, line_dash="dash", 
                     line_color="red", annotation_text="Your Age")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in population comparison: {e}")
        st.warning("Could not generate population comparison.")

# --- Main App ---
def main():
    """Main application function."""
    load_css()

    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Risk AI 🩺</h1>', unsafe_allow_html=True)
    st.markdown("#### An AI-powered tool to estimate cardiovascular disease risk using advanced machine learning models")

    # --- Load Data ---
    stats_df = load_mock_statistics()

    # --- Model Selection ---
    selected_model_key, selected_model_info = create_model_selector()

    # --- Sidebar ---
    with st.sidebar:
        # Using a heart emoji instead of external image
        st.markdown("# ❤️")
        st.subheader("Available Models")
        
        for key, info in AVAILABLE_MODELS.items():
            is_selected = "✅" if key == selected_model_key else "⚪"
            st.markdown(f"""
            {is_selected} **{info['name']}**
            - *{info['type']}*
            - {info['description']}
            """)
        
        st.markdown("---")
        st.subheader("Features")
        st.info("""
        **Input Features:** 
        - Age, Gender, Height, Weight
        - Blood pressure (systolic/diastolic)
        - Cholesterol and glucose levels
        - Lifestyle factors (smoking, alcohol, activity)
        
        **Advanced Processing:**
        - Feature engineering
        - Risk score calculation
        - Population comparison
        
        **Note:** If V4 has compatibility issues, a simplified prediction method will be used automatically.
        """)

    # --- Load Selected Model ---
    with st.spinner(f"Loading {selected_model_info['name']}..."):
        model_data = load_model(
            selected_model_info['repo_id'], 
            selected_model_info['filename'],
            selected_model_key
        )

    # --- App Layout ---
    if not model_data:
        st.error(f"⚠️ {selected_model_info['name']} could not be loaded. Please check your internet connection and try again.")
        st.info("The application requires access to the pre-trained model to function properly.")
        return

    # Debug model structure for V4
    if "v4" in selected_model_key and model_data:
        debug_model_structure(model_data, selected_model_key)

    # Medical Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool provides risk estimates for educational purposes only and is <strong>not a substitute</strong> for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.
    </div>
    """, unsafe_allow_html=True)

    # Input Form
    patient_data_input = create_input_form()
    
    # Analysis Button
    if st.button("🔍 Analyze My Cardiovascular Risk", type="primary", use_container_width=True):
        # Validate input data
        validation_errors = validate_patient_data(patient_data_input)
        
        if validation_errors:
            st.error("⚠️ Please correct the following issues:")
            for error in validation_errors:
                st.error(f"• {error}")
        else:
            with st.spinner(f"Analyzing your cardiovascular risk profile using {selected_model_info['name']}..."):
                try:
                    # Feature Engineering - pass model_data for V4
                    engineered_df = create_feature_engineering(patient_data_input.copy(), selected_model_key, model_data)
                    
                    if engineered_df is not None:
                        # Make Prediction
                        prediction_result = make_prediction(model_data, engineered_df, selected_model_info, patient_data_input)
                        
                        if prediction_result:
                            # Display Results
                            display_results(prediction_result)
                            display_risk_breakdown_and_recommendations(patient_data_input, engineered_df)
                            
                            # Population comparison (if data available)
                            if stats_df is not None:
                                display_population_comparison(patient_data_input, stats_df)
                                
                            # Additional information
                            with st.expander("📊 Understanding Your Results"):
                                st.write(f"""
                                **How to interpret your results from {selected_model_info['name']}:**
                                
                                - **Low Risk (0-30%)**: Excellent cardiovascular health profile
                                - **Moderate Risk (30-60%)**: Some risk factors present, lifestyle changes recommended
                                - **High Risk (60-80%)**: Multiple risk factors, medical consultation advised
                                - **Critical Risk (80-100%)**: Immediate medical attention recommended
                                
                                **Model Information:**
                                - **Type:** {selected_model_info['type']}
                                - **Description:** {selected_model_info['description']}
                                
                                **Important Notes:**
                                - This assessment is based on major cardiovascular risk factors
                                - Individual risk may vary based on family history and other factors not captured here
                                - Regular health check-ups are important regardless of risk level
                                - Different models may provide slightly different risk estimates
                                
                                **Technical Notes:**
                                - If using V4 simplified method: The prediction uses a rule-based approach when the advanced model encounters compatibility issues
                                - Both methods provide clinically relevant risk assessments based on established cardiovascular risk factors
                                """)
                                
                            # Model comparison suggestion
                            if selected_model_key != "heart_risk_ai_v4":
                                st.info("💡 **Tip:** Try analyzing with different models to compare results and gain additional insights!")
                                
                        else:
                            st.error("❌ Could not generate risk assessment. Please try again.")
                    else:
                        st.error("❌ Error processing your health data. Please check your inputs and try again.")
                        
                except Exception as e:
                    logger.error(f"Unexpected error during analysis: {e}")
                    st.error("❌ An unexpected error occurred during analysis. Please try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Made with ❤️ for cardiovascular health awareness by <strong>Juan Manuel Infante Quiroga</strong><br>
        Powered by advanced AI models for accurate risk assessment<br>
        Always consult healthcare professionals for medical decisions
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()