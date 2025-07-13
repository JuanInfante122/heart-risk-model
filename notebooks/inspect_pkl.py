import joblib

# Load the content of the .pkl file
pkl_content = joblib.load('C:\Dev\heart-risk-ai\models\heart_risk_ensemble.pkl')

# Print the content type and the keys (if it's a dictionary)
print(f"Content type: {type(pkl_content)}")
if isinstance(pkl_content, dict):
    print(f"Dictionary keys: {pkl_content.keys()}")
