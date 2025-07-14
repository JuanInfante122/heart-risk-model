# ❤️ Heart Risk AI - Advanced Cardiovascular Risk Predictor

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg) ![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)

This project provides an advanced AI-powered tool to predict cardiovascular risk. It includes a full MLOps pipeline, from data preprocessing and advanced feature engineering to a real-time interactive web application that allows users to choose between two different model versions.

## Features

- **Dual Model Support:** Switch between a baseline model and an advanced, fine-tuned version.
- **Interactive UI:** A user-friendly Streamlit application for real-time predictions.
- **Advanced Machine Learning:** The v2 model uses a sophisticated stacked ensemble, advanced feature engineering, and probability calibration.
- **Detailed Analysis:** The app provides a breakdown of risk factors and personalized recommendations.

## Quick Start & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JuanInfante122/heart-risk-model.git
    cd heart-risk-model
    ```

2.  **Set up the environment:**
    ```bash
    make setup
    make install
    ```

3.  **Configure Hugging Face Token:**
    This application loads models from a private Hugging Face repository. You need to provide an access token.

    - Create a file at `.streamlit/secrets.toml`.
    - Add your Hugging Face token to this file as follows:
      ```toml
      HUGGING_FACE_HUB_TOKEN = "hf_YOUR_TOKEN_HERE"
      ```

4.  **Run the application:**
    ```bash
    make run
    ```

## Model Evolution & Results

This project showcases a clear evolution from a baseline model to a more advanced and robust predictor. Below is a summary of the results for both versions.

| Metric      | Baseline Model (v1) | Advanced Model (v2) |
|-------------|-----------------------|-----------------------|
| **AUC-ROC**     | 80.0%                 | **79.4%**             |
| **Sensitivity** | 85.0%                 | **85.2%**             |
| **Specificity** | 55.0%                 | **53.4%**             |
| **F1-Score**    | -                     | **73.5%**             |
| **Accuracy**    | -                     | **69.3%**             |

*The v2 model was optimized to maintain high sensitivity while improving overall performance and robustness.*

## Methodology

- **Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle (70,000 entries).
- **Goal:** Binary classification to predict 10-year cardiovascular risk.

### Baseline Model (v1)

- **Models:** A simple ensemble of Random Forest and XGBoost.
- **Feature Engineering:** Basic features derived from the original dataset.
- **Validation:** Standard 80/20 stratified split.

### Advanced Model (v2)

The second version of the model represents a significant leap in methodology:

- **Advanced Feature Engineering:** Includes polynomial features, statistical aggregations, and medical risk scores (e.g., Framingham, pulse pressure).
- **Intelligent Feature Selection:** Combines multiple techniques (Univariate stats, RFE, Tree-based importance, L1 regularization) to select the most impactful features.
- **Advanced Sampling:** Evaluates multiple strategies (SMOTE, ADASYN, etc.) to handle class imbalance, selecting the best-performing one for training.
- **Hyperparameter Tuning:** Uses Optuna for sophisticated and efficient hyperparameter optimization.
- **Stacked Ensemble:** Creates a powerful stacked ensemble model using the best-performing base models as estimators and a logistic regression model as the meta-learner.
- **Probability Calibration:** Applies isotonic regression to calibrate model probabilities, making them more reliable.

## Tech Stack

- Python 3.8+
- Scikit-learn, XGBoost, LightGBM, CatBoost
- Pandas, NumPy
- Streamlit for the interactive web app
- Optuna for hyperparameter tuning

## Medical Disclaimer

This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

## Contributing

Contributions are welcome! Please follow the guidelines in `CONTRIBUTING.md`.

## License

MIT License (see LICENSE).
