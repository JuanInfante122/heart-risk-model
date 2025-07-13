---
license: mit
tags:
- sklearn
- ensemble
- heart-disease
- cardiovascular-risk
- imbalanced-data
- explainable-ai
metrics:
- roc_auc
- accuracy
- recall
- precision
- f1
---

# Model Card for Heart Risk AI - Cardiovascular Risk Prediction v3

This is an advanced ensemble model designed to predict the risk of cardiovascular disease, combining Random Forest and XGBoost algorithms. The model is optimized for high sensitivity to minimize missed diagnoses in medical screening scenarios.

## Model Details

### Model Description

This model combines two powerful machine learning algorithms (Random Forest and XGBoost) in an ensemble approach to predict cardiovascular disease risk. The model is specifically optimized for high sensitivity (recall), making it suitable for initial screening where detecting all potential cases is prioritized over minimizing false positives. It uses comprehensive feature engineering on the Kaggle Cardiovascular Disease Dataset.

- **Developed by:** Juan Manuel Infante Quiroga
- **Model type:** Ensemble Classification (Random Forest + XGBoost)
- **Language(s):** Python (scikit-learn, XGBoost)
- **License:** Apache 2.0
- **Version:** v3 (optimized for high sensitivity)

### Model Sources

- **Repository:** Juan12Dev/heart-risk-ai
- **Dataset:** Kaggle Cardiovascular Disease Dataset
- **Framework:** scikit-learn, XGBoost

## Uses

### Direct Use

This model is intended for **educational and research purposes only**. It can be used as a screening tool for:

- Initial cardiovascular risk assessment in general population
- Patient education and awareness about cardiovascular risk factors
- Research studies on cardiovascular risk prediction
- Educational platforms for medical training

**⚠️ IMPORTANT: This tool does NOT replace complete clinical evaluation, specific diagnostic tests, or professional medical judgment.**

### Downstream Use

The model can be integrated into:
- Healthcare screening applications
- Clinical decision support systems (as a supplementary tool)
- Research studies on cardiovascular risk prediction
- Educational platforms for medical training

### Out-of-Scope Use

- **Definitive diagnosis:** Never use for final diagnostic decisions
- **Treatment decisions:** Not suitable for determining treatment protocols
- **Standalone medical advice:** Requires professional medical interpretation
- **Real-time emergency decisions:** Not validated for acute care settings

## Bias, Risks, and Limitations

### Known Limitations

1. **Dataset Origin:** The dataset's specific geographic and demographic origins are not fully detailed, which may influence the model's performance on different populations
2. **Data Quality:** The data is self-reported and may contain inaccuracies
3. **Temporal Validation:** The model requires validation in a prospective clinical cohort
4. **High False Positive Rate:** 45.3% of healthy patients may be incorrectly flagged as at-risk
5. **Feature Engineering Dependency:** Model relies on complex feature engineering that must be replicated exactly

### Technical Risks

- **Low Specificity (54.7%):** High number of false positives may cause unnecessary anxiety and healthcare costs
- **Self-reported Data:** Original dataset relies on patient-reported information which may introduce bias
- **Feature Engineering Complexity:** Requires exact replication of 15+ engineered features for proper functioning

### Recommendations

- Use only as a preliminary screening tool, never for final diagnosis
- Always combine with clinical judgment and additional testing
- Consider the high false positive rate when interpreting results
- Validate performance on diverse populations before broader deployment
- Implement appropriate patient counseling protocols for positive predictions
- Ensure exact feature engineering replication when implementing

## How to Get Started with the Model

```python
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# Load the model from Hugging Face
model_path = hf_hub_download(repo_id="Juan12Dev/heart-risk-ai", filename="heart_risk_ensemble_v3.pkl")
model_data = joblib.load(model_path)

ensemble_model = model_data['ensemble_model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
optimal_threshold = model_data['optimal_threshold']

# Example patient data (must match the Kaggle dataset structure)
patient_data = {
    'age': 50,          # Age in years
    'gender': 2,        # 1: female, 2: male
    'height': 168,      # Height in cm
    'weight': 70.0,     # Weight in kg
    'ap_hi': 120,       # Systolic blood pressure
    'ap_lo': 80,        # Diastolic blood pressure
    'cholesterol': 1,   # 1: normal, 2: above normal, 3: well above normal
    'gluc': 1,          # 1: normal, 2: above normal, 3: well above normal
    'smoke': 0,         # 0: no, 1: yes
    'alco': 0,          # 0: no, 1: yes
    'active': 1         # 0: no, 1: yes (physical activity)
}

# Feature Engineering (must be identical to training)
def create_feature_engineering(patient_data):
    df = pd.DataFrame([patient_data])
    
    # Age conversions and risk calculations
    df['age_in_days'] = df['age'] * 365.25
    df.rename(columns={'age_in_days': 'age'}, inplace=True)
    df['age_group'] = pd.cut(df['age']/365.25, bins=[0, 45, 55, 65, 100], 
                            labels=['<45', '45-55', '55-65', '65+'])
    df['age_normalized'] = (df['age'] - (25 * 365.25)) / ((70*365.25) - (25*365.25))
    df['age_risk_exponential'] = np.where(df['age']/365.25 > 45, 
                                         np.exp((df['age']/365.25 - 45) / 10), 1.0)
    
    # Blood pressure features
    df['bp_category'] = pd.cut(df['ap_hi'], bins=[0, 120, 140, 160, 180, 1000], 
                              labels=['Normal', 'Elevated', 'Hypertension Stage 1', 
                                     'Hypertension Stage 2', 'Hypertensive Crisis'])
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Metabolic features
    df['metabolic_profile'] = df['cholesterol'] / (df['age']/365.25)
    df['chol_category'] = pd.cut(df['cholesterol'], bins=[0, 1, 2, 3, 1000], 
                                labels=['Normal', 'Above Normal', 'Well Above Normal', 'High'])
    df['metabolic_syndrome_risk'] = ((df['cholesterol'] > 1).astype(int) + 
                                    (df['gluc'] > 1).astype(int) + 
                                    (df['ap_hi'] > 140).astype(int))
    
    # Gender interactions
    df['male_age_interaction'] = df['gender'] * (df['age']/365.25)
    df['female_chol_interaction'] = (1 - df['gender']) * df['cholesterol']
    df['gender_specific_risk'] = np.where(df['gender'] == 2, 
                                         (df['age']/365.25) * 0.1 + df['cholesterol'] * 0.005, 
                                         df['cholesterol'] * 0.008)
    
    # Risk scores
    df['traditional_risk_score'] = (df['age']/365.25 * 0.04 + df['gender'] * 10 + 
                                   (df['cholesterol'] - 1) * 20 + df['ap_hi'] * 0.1 + 
                                   df['gluc'] * 20)
    df['cardiac_risk_score'] = (df['pulse_pressure'] * 0.2 + df['ap_hi'] * 0.1)
    df['combined_risk_score'] = (df['traditional_risk_score'] * 0.4 + 
                                df['cardiac_risk_score'] * 0.6)
    
    # Categorical encodings
    for col in ['age_group', 'chol_category', 'bp_category']:
        categories = ['<45', '45-55', '55-65', '65+', 'Normal', 'Elevated', 
                     'Hypertension Stage 1', 'Hypertension Stage 2', 
                     'Hypertensive Crisis', 'Above Normal', 'Well Above Normal', 'High']
        df[f'{col}_encoded'] = pd.Categorical(df[col], categories=categories).codes
    
    return df

# Apply feature engineering
engineered_df = create_feature_engineering(patient_data)
X = engineered_df.reindex(columns=feature_names, fill_value=0)
X_scaled = scaler.transform(X)

# Make prediction
probability = ensemble_model.predict_proba(X_scaled)[0, 1]
prediction = (probability >= optimal_threshold).astype(int)

print(f"Probability of Heart Disease: {probability:.2f}")
print(f"Prediction (1=Disease, 0=No Disease): {prediction}")
```

## Training Details

### Training Data

- **Source:** Kaggle Cardiovascular Disease Dataset
- **Size:** 70,000 patient records
- **Features:** 11 base features + 15+ engineered features
- **Base Features:** age, gender, height, weight, ap_hi (systolic BP), ap_lo (diastolic BP), cholesterol, glucose, smoke, alcohol, physical activity
- **Target:** Binary classification (presence/absence of heart disease)
- **Preprocessing:** SMOTE (Synthetic Minority Over-sampling Technique) applied to handle class imbalance

### Training Procedure

#### Preprocessing

1. Comprehensive feature engineering creating 15+ additional predictive features
2. Feature scaling using StandardScaler
3. SMOTE oversampling for class balance
4. Train/test split: 80/20 (56,000 train / 14,000 test)

#### Training Hyperparameters

- **Training regime:** Standard precision (fp32)
- **Ensemble method:** Combination of Random Forest and XGBoost
- **Cross-validation:** Stratified cross-validation
- **Optimization target:** Maximized sensitivity (recall)
- **Threshold optimization:** Custom threshold tuning for high sensitivity

#### Speeds, Sizes, Times

- **Training time:** 2-4 hours on standard hardware
- **Model size:** ~15 MB
- **Inference time:** < 100ms per prediction

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **Size:** 20% holdout set (14,000 samples) from the Kaggle dataset
- **Evaluation method:** Stratified sampling to maintain class distribution

#### Factors

- **Age groups:** Tested across different age ranges
- **Gender:** Performance evaluated for both male and female patients
- **Risk levels:** Evaluated across different cardiovascular risk profiles

#### Metrics

Performance optimized for medical screening scenarios where high sensitivity is critical:

- **ROC-AUC:** Area under the receiver operating characteristic curve
- **Sensitivity (Recall):** True positive rate - primary optimization target
- **Specificity:** True negative rate
- **Precision:** Positive predictive value
- **F1-Score:** Harmonic mean of precision and recall

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | 0.7986 | Good discriminative ability |
| **Accuracy** | 0.6984 | Overall correct predictions |
| **Sensitivity (Recall)** | 0.8503 ⭐ | Excellent - catches 85% of actual cases |
| **Specificity** | 0.5467 | Moderate - 45% false positive rate |
| **Precision** | 0.6520 | When model says "high risk", it's correct 65% of time |
| **F1-Score** | 0.7381 | Balanced performance measure |

#### Summary

The model achieves its primary objective of high sensitivity (85.03%) with strong statistical reliability due to training on 70,000 samples. This makes it highly effective for screening applications where missing a positive case is more costly than having false positives. The large training dataset provides confidence in the model's generalizability, though the trade-off remains a high false positive rate (45.3%), requiring careful implementation with appropriate follow-up protocols.

## Model Examination

The ensemble approach provides interpretability through:
- Feature importance rankings from Random Forest
- SHAP values computation for individual predictions
- Individual model predictions for transparency
- Feature engineering transparency with documented transformations

## Environmental Impact

- **Hardware Type:** Standard CPU (no GPU required)
- **Training Time:** 2-4 hours on standard hardware (with 70K samples)
- **Model Size:** ~15 MB
- **Inference:** Real-time prediction capability
- **Carbon Footprint:** Low due to efficient algorithms and CPU-only training

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** Ensemble of Random Forest and XGBoost with optimized voting
- **Objective:** Binary classification optimized for maximum sensitivity
- **Input:** 11 base features + 15+ engineered features
- **Output:** Risk probability + binary classification with optimized threshold

### Compute Infrastructure

#### Hardware

- Standard CPU sufficient for training and inference
- Memory requirements: 2-4 GB RAM for training, < 1 GB for inference
- No specialized hardware needed

#### Software

- Python 3.7+
- scikit-learn
- XGBoost
- pandas, numpy
- joblib for model serialization
- huggingface_hub for model distribution

## Citation

**BibTeX:**

```bibtex
@misc{heart_risk_ai_v3_2024,
  title={Heart Risk AI: Cardiovascular Risk Prediction Model v3},
  author={Juan Manuel Infante Quiroga},
  year={2024},
  note={Ensemble model optimized for high sensitivity in cardiovascular screening},
  howpublished={Hugging Face Hub},
  url={https://huggingface.co/Juan12Dev/heart-risk-ai}
}
```

**APA:**

Infante Quiroga, J. M. (2024). *Heart Risk AI: Cardiovascular Risk Prediction Model v3*. Hugging Face Hub. https://huggingface.co/Juan12Dev/heart-risk-ai

## Glossary

- **Sensitivity (Recall):** Proportion of actual positive cases correctly identified
- **Specificity:** Proportion of actual negative cases correctly identified  
- **SMOTE:** Synthetic Minority Over-sampling Technique for handling imbalanced data
- **Ensemble:** Combination of multiple models to improve prediction performance
- **ROC-AUC:** Receiver Operating Characteristic - Area Under Curve, measures overall discriminative ability
- **Feature Engineering:** Process of creating new predictive features from existing data
- **Pulse Pressure:** Difference between systolic and diastolic blood pressure

## Model Card Authors

Juan Manuel Infante Quiroga

## Model Card Contact

Juan Manuel Infante Quiroga - Available through Hugging Face Hub