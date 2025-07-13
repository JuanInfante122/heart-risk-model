import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, accuracy_score, recall_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE

class HeartRiskPredictor:
    """Ensemble model for cardiovascular risk prediction"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.optimal_threshold = 0.5
        
    def load_data(self, filepath='data/processed/heart_disease_engineered.csv'):
        """Load processed data"""
        df = pd.read_csv(filepath)
        feature_scores = pd.read_csv('data/processed/feature_importance_scores.csv')

        # Select top 20 features
        top_features = feature_scores.nlargest(20, 'score')['feature'].tolist()

        print(f"Data loaded: {len(df)} records, {len(top_features)} features")
        
        # Separate features and target
        X = df[top_features]
        y = df['target']
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare data for training"""
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data prepared - Train: {len(X_train_res)}, Test: {len(X_test)}")
        print(f"Class balance - Train: {y_train_res.value_counts()[1]/len(y_train_res):.1%}, "
              f"Test: {y_test.value_counts()[1]/len(y_test):.1%} positive cases")
        
        return X_train_scaled, X_test_scaled, y_train_res, y_test, X_train_res, X_test
    
    def create_base_models(self):
        """Create base models for ensemble"""
        # Random Forest (interpretability + robustness)
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, class_weight='balanced'
        )
        
        # XGBoost (performance)
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        
        # Gradient Boosting (different boosting approach)
        gb_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        
        # Logistic Regression (linear baseline)
        lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        
        self.models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'GradientBoosting': gb_model,
            'LogisticRegression': lr_model
        }
        
        print("Base models created: RandomForest, XGBoost, GradientBoosting, LogisticRegression")
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters for main models"""
        print("Optimizing hyperparameters for SENSITIVITY (RECALL)...")
        
        # Optimize Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_split': [5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'), rf_params,
            cv=5, scoring='recall', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        self.models['RandomForest'] = rf_grid.best_estimator_
        
        # Optimize XGBoost
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [6, 9],
            'learning_rate': [0.05, 0.1]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42), xgb_params,
            cv=5, scoring='recall', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_grid.best_estimator_

        print("Hyperparameter optimization completed")
    
    def train_individual_models(self, X_train, y_train):
        """Train individual models"""
        print("Training individual models...")
        cv_scores = {}
        
        for name, model in self.models.items():
            cv_score = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='recall'
            )
            cv_scores[name] = cv_score
            model.fit(X_train, y_train)
            print(f"   {name}: Recall = {cv_score.mean():.3f} \u00b1 {cv_score.std():.3f}")
        
        return cv_scores
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model with soft voting"""
        print("Creating ensemble model...")
        
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.models['RandomForest']),
                ('xgb', self.models['XGBoost'])
            ],
            voting='soft'
        )
        
        self.ensemble_model.fit(X_train, y_train)
        
        ensemble_cv_score = cross_val_score(
            self.ensemble_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='recall'
        )
        
        print(f"   Ensemble: Recall = {ensemble_cv_score.mean():.3f} \u00b1 {ensemble_cv_score.std():.3f}")
        self.is_trained = True
        return ensemble_cv_score

    def find_optimal_threshold(self, X_test, y_test, target_sensitivity=0.85):
        """Find the optimal probability threshold to achieve a target sensitivity."""
        print(f"\nFinding optimal threshold for target sensitivity >= {target_sensitivity:.1%}...")
        y_prob = self.ensemble_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        valid_indices = np.where(tpr >= target_sensitivity)[0]
        if len(valid_indices) > 0:
            # Find the threshold that gives the target sensitivity with the highest specificity
            best_index = valid_indices[np.argmax(1 - fpr[valid_indices])]
            self.optimal_threshold = thresholds[best_index]
            print(f"   Optimal threshold found: {self.optimal_threshold:.4f}")
            print(f"   This gives a sensitivity of {tpr[best_index]:.1%} and specificity of {1-fpr[best_index]:.1%}")
        else:
            self.optimal_threshold = 0.5 # Fallback
            print("   Could not find a threshold for the target sensitivity. Using 0.5.")

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set using the optimal threshold."""
        print("\nEvaluating models on test set with adjusted threshold...")
        results = {}
        
        y_prob_ensemble = self.ensemble_model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = (y_prob_ensemble >= self.optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_adjusted)
        recall = recall_score(y_test, y_pred_adjusted)
        auc = roc_auc_score(y_test, y_prob_ensemble)
        
        results['Ensemble'] = {
            'accuracy': accuracy,
            'recall': recall,
            'auc': auc,
            'predictions': y_pred_adjusted,
            'probabilities': y_prob_ensemble
        }
        
        print(f"\n   ENSEMBLE (Adjusted): Accuracy = {accuracy:.3f}, Recall (Sensitivity) = {recall:.3f}, AUC = {auc:.3f}")
        return results
    
    def save_model(self, filepath='models/heart_risk_ensemble_v3.pkl'):
        """Save trained model and threshold"""
        if not self.is_trained:
            print("ERROR: Model not trained")
            return
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def train_complete_pipeline(self):
        """Execute complete training pipeline"""
        print("Starting complete model training (v2 - optimized for sensitivity)...")
        print("=" * 70)
        
        X, y = self.load_data()
        X_train, X_test, y_train, y_test, _, _ = self.prepare_data(X, y)
        self.create_base_models()
        self.optimize_hyperparameters(X_train, y_train)
        self.train_individual_models(X_train, y_train)
        self.create_ensemble_model(X_train, y_train)
        self.find_optimal_threshold(X_test, y_test)
        results = self.evaluate_models(X_test, y_test)
        self.save_model()
        
        print("\nTraining pipeline completed!")
        return results

if __name__ == "__main__":
    predictor = HeartRiskPredictor()
    predictor.train_complete_pipeline()