import unittest
import pandas as pd
from train_model.feature_engineering import HeartRiskFeatureEngineer

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataframe for testing."""
        self.engineer = HeartRiskFeatureEngineer()
        self.df = pd.DataFrame({
            'age': [52, 57, 58, 60, 62],
            'sex': [1, 0, 1, 1, 0],
            'cp': [1, 2, 2, 3, 4],
            'trestbps': [125, 130, 132, 140, 150],
            'chol': [212, 236, 224, 253, 260],
            'fbs': [0, 0, 1, 0, 1],
            'restecg': [1, 0, 0, 1, 1],
            'thalach': [168, 174, 173, 144, 150],
            'exang': [0, 0, 0, 1, 0],
            'oldpeak': [1.0, 0.0, 3.2, 1.4, 0.8],
            'slope': [2, 2, 2, 1, 1],
            'ca': [2, 3, 2, 1, 0],
            'thal': [3, 2, 3, 3, 2],
            'target': [0, 1, 0, 0, 1]
        })

    def test_load_data(self):
        """Test that the data is loaded correctly."""
        # For this test, we'll use the actual data file
        df_loaded = self.engineer.load_data(filepath='C:/Dev/heart-risk-ai/data/raw/heart_disease_uci.csv')
        self.assertIsInstance(df_loaded, pd.DataFrame)
        self.assertGreater(len(df_loaded), 0)

    def test_create_age_features(self):
        """Test the creation of age-related features."""
        df_featured = self.engineer.create_age_features(self.df.copy())
        self.assertIn('age_group', df_featured.columns)
        self.assertIn('age_normalized', df_featured.columns)
        self.assertIn('age_risk_exponential', df_featured.columns)

if __name__ == '__main__':
    unittest.main()
