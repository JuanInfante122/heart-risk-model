# ❤️ Heart Risk AI - 80% AUC Cardiovascular Risk Predictor

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg) ![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)

## Highlights

- **80% AUC-ROC**
- **85% Sensitivity** (catches real cases)
- **55% Specificity** (avoids false positives)
- Ensemble of 2 models: Random Forest & XGBoost
- Interactive real-time web app using Streamlit
- Full MLOps pipeline from preprocessing to deployment

## Quick Start

```bash
git clone https://github.com/Juan12Dev/heart-risk-ai.git
cd heart-risk-ai
make setup
make install
make run
```

## Results Summary

| Metric      | Value   |
|-------------|---------|
| AUC-ROC     | 80%   |
| Sensitivity | 85%   |
| Specificity | 55%   |

## Screenshot

*Add a working screenshot of the Streamlit interface.*

## Methodology

- **Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle (70,000 entries)
- **Goal:** Binary classification - cardiovascular risk

### Models

- **Random Forest:** Feature robustness
- **XGBoost:** High-performance boosting
- **Gradient Boosting:** Sequential learning
- **Logistic Regression:** Interpretable baseline
- **Ensemble:** Soft voting classifier

### Validation

- 80/20 stratified split
- Hyperparameter tuning via cross-validation
- Results compared against medical benchmarks

## Tech Stack

- Python 3.8+
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Matplotlib / Seaborn
- Streamlit

## Live Demo

[Insert Streamlit demo link here]

## Documentation

- Setup Guide
- Model Documentation
- Results & Insights
- Deployment Instructions

## Medical Disclaimer

This tool is for educational purposes only. It is not medical advice and should not replace consultation with a licensed professional.

## Contributing

Contributions welcome! Follow CONTRIBUTING.md.

## License

MIT License (see LICENSE)

## Acknowledgments

- Kaggle for providing the dataset.
- The open-source libraries and contributors that made this project possible.
- Kaggle for providing the dataset.
- The open-source libraries and contributors that made this project possible.

⭐ Don’t forget to star the repo if you find it useful!