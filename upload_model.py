import joblib
import skops.hub_utils as hub_utils
import skops.card as card
from skops.io import dump
import tempfile
from pathlib import Path
import pandas as pd
from huggingface_hub import create_repo, HfApi
import os

# --- 1. Load your trained model and data ---
model_path = 'models/heart_risk_ensemble_v3.pkl'
model_data = joblib.load(model_path)
model = model_data['ensemble_model']
data = pd.read_csv('data/processed/heart_disease_engineered.csv')
X = data[model_data['feature_names']]

# --- 2. Create a temporary directory and save the model ---
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_model_path = Path(tmpdir) / 'heart_risk_ensemble_v3.skops'
    dump(model, tmp_model_path)

    # --- 3. Create a local repository ---
    local_repo = Path(tmpdir) / "repo"
    local_repo.mkdir()

    # --- 4. Initialize the repository ---
    hub_utils.init(
        model=tmp_model_path,
        requirements=['scikit-learn', 'skops'],
        dst=local_repo,
        task='tabular-classification',
        data=X,
    )

    # --- 5. Create a model card ---
    model_card = card.Card(model)
    model_card.save(local_repo / 'README.md')

    # --- 6. Push the repository to the Hub ---
    # Replace with your Hugging Face username and desired repo name
    repo_id = 'Damn-Yuansh/heart-risk-ai'
    
    # The token will be automatically picked up from the HF_TOKEN environment variable
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    hub_utils.push(
        repo_id=repo_id,
        source=local_repo,
        commit_message='Upload heart_risk_ensemble_v3 model',
    )

    print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")
