import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                   StratifiedKFold, validation_curve, learning_curve)
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, accuracy_score, recall_score,
                            precision_score, f1_score, brier_score_loss, log_loss,
                            average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import optuna
from scipy import stats

class AdvancedHeartRiskPredictor:
    """
    Advanced Ensemble Model for Cardiovascular Risk Prediction
    
    Features:
    - Advanced feature engineering and selection
    - Multiple sampling strategies
    - Hyperparameter optimization with Optuna
    - Probability calibration
    - Comprehensive evaluation and visualization
    - Model interpretation tools
    """
    
    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        
        # Model components
        self.models = {}
        self.ensemble_model = None
        self.calibrated_ensemble = None
        self.scaler = None
        self.feature_selector = None
        
        # Training artifacts
        self.feature_names = []
        self.feature_importance_df = None
        self.training_history = {}
        self.calibration_curve_data = {}
        
        # Model parameters
        self.optimal_threshold = 0.5
        self.is_trained = False
        self.target_sensitivity = 0.85
        
        # Results storage
        self.results = {}
        self.cv_results = {}
        
        # Create results directory
        self.results_dir = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        if self.verbose:
            print("🚀 Advanced Heart Risk Predictor Initialized")
            print(f"📁 Results will be saved to: {self.results_dir}")
            print("=" * 80)
    
    def load_and_validate_data(self, filepath='data/processed/heart_disease_engineered.csv'):
        """Load and validate training data with comprehensive analysis"""
        print("\n📊 LOADING AND VALIDATING DATA")
        print("-" * 50)
        
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Data loaded successfully: {len(df)} records")
        except FileNotFoundError:
            print(f"❌ Error: File {filepath} not found")
            return None, None
        
        # Data validation and quality checks
        print(f"   📏 Dataset shape: {df.shape}")
        print(f"   🎯 Target distribution: {df['target'].value_counts().to_dict()}")
        print(f"   📈 Positive rate: {df['target'].mean():.1%}")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"   ⚠️  Missing values found: {missing_data[missing_data > 0].to_dict()}")
        else:
            print("   ✅ No missing values found")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"   ⚠️  Duplicate rows found: {duplicates}")
            df = df.drop_duplicates()
            print(f"   🧹 Duplicates removed. New shape: {df.shape}")
        
        # Statistical summary
        print(f"   📊 Numerical features: {df.select_dtypes(include=[np.number]).shape[1]}")
        print(f"   🏷️  Categorical features: {df.select_dtypes(include=['object']).shape[1]}")
        
        # Feature importance from previous analysis (if available)
        try:
            feature_scores = pd.read_csv('data/processed/feature_importance_scores.csv')
            top_features = feature_scores.nlargest(25, 'score')['feature'].tolist()
            print(f"   🎯 Using top {len(top_features)} features from previous analysis")
        except FileNotFoundError:
            # Use all numeric features except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            top_features = [col for col in numeric_cols if col != 'target']
            print(f"   📋 Using all {len(top_features)} numeric features")
        
        # Separate features and target
        X = df[top_features]
        y = df['target']
        
        # Additional data quality checks
        self._perform_data_quality_checks(X, y)
        
        self.feature_names = X.columns.tolist()
        print(f"   ✅ Final feature set: {len(self.feature_names)} features")
        
        return X, y
    
    def _perform_data_quality_checks(self, X, y):
        """Perform comprehensive data quality checks"""
        print("\n🔍 DATA QUALITY ANALYSIS")
        
        # Check for high correlation features
        corr_matrix = X.corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"   ⚠️  High correlation pairs found: {len(high_corr_pairs)}")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # Show first 5
                print(f"      {feat1} ↔ {feat2}: {corr:.3f}")
        
        # Check for features with low variance
        low_variance_features = X.columns[X.var() < 0.01].tolist()
        if low_variance_features:
            print(f"   ⚠️  Low variance features: {low_variance_features}")
        
        # Check class balance
        class_balance = y.value_counts(normalize=True)
        imbalance_ratio = class_balance.min() / class_balance.max()
        print(f"   ⚖️  Class balance ratio: {imbalance_ratio:.3f}")
        if imbalance_ratio < 0.3:
            print("   📢 Dataset is imbalanced - will apply balancing techniques")
    
    def advanced_feature_engineering(self, X, y):
        """Apply advanced feature engineering techniques"""
        print("\n🔧 ADVANCED FEATURE ENGINEERING")
        print("-" * 50)
        
        X_engineered = X.copy()
        original_features = len(X_engineered.columns)
        
        # 1. Polynomial features for key interactions
        print("   🧮 Creating polynomial interaction features...")
        from sklearn.preprocessing import PolynomialFeatures
        key_features = ['age', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        available_key_features = [f for f in key_features if f in X_engineered.columns]
        
        if len(available_key_features) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            key_data = X_engineered[available_key_features]
            poly_features = poly.fit_transform(key_data)
            poly_feature_names = poly.get_feature_names_out(available_key_features)
            
            # Add only new interaction features (not original features)
            new_features = poly_features[:, len(available_key_features):]
            new_feature_names = poly_feature_names[len(available_key_features):]
            
            for i, name in enumerate(new_feature_names):
                X_engineered[f'poly_{name}'] = new_features[:, i]
        
        # 2. Statistical aggregations
        print("   📈 Creating statistical aggregation features...")
        if len(available_key_features) >= 3:
            feature_data = X_engineered[available_key_features]
            X_engineered['feature_mean'] = feature_data.mean(axis=1)
            X_engineered['feature_std'] = feature_data.std(axis=1)
            X_engineered['feature_median'] = feature_data.median(axis=1)
            X_engineered['feature_max'] = feature_data.max(axis=1)
            X_engineered['feature_min'] = feature_data.min(axis=1)
            X_engineered['feature_range'] = X_engineered['feature_max'] - X_engineered['feature_min']
        
        # 3. Risk scores based on medical knowledge
        print("   🏥 Creating medical risk scores...")
        if all(f in X_engineered.columns for f in ['age', 'ap_hi', 'cholesterol']):
            X_engineered['framingham_score'] = (
                X_engineered['age'] * 0.04 + 
                (X_engineered['ap_hi'] - 120) * 0.02 + 
                X_engineered['cholesterol'] * 15
            )
        
        if all(f in X_engineered.columns for f in ['ap_hi', 'ap_lo']):
            X_engineered['pulse_pressure'] = X_engineered['ap_hi'] - X_engineered['ap_lo']
            X_engineered['mean_arterial_pressure'] = X_engineered['ap_lo'] + (X_engineered['pulse_pressure'] / 3)
        
        # 4. Age-based features
        if 'age' in X_engineered.columns:
            X_engineered['age_squared'] = X_engineered['age'] ** 2
            X_engineered['age_log'] = np.log1p(X_engineered['age'])
            X_engineered['age_risk_category'] = pd.cut(X_engineered['age'], 
                                                     bins=[0, 45, 55, 65, 100], 
                                                     labels=[0, 1, 2, 3]).astype(float)
        
        print(f"   ✅ Feature engineering completed: {original_features} → {len(X_engineered.columns)} features")
        return X_engineered
    
    def intelligent_feature_selection(self, X, y):
        """Apply multiple feature selection techniques and combine results"""
        print("\n🎯 INTELLIGENT FEATURE SELECTION")
        print("-" * 50)
        
        # Prepare data
        X_scaled = StandardScaler().fit_transform(X)
        
        feature_scores = {}
        
        # 1. Univariate statistical tests
        print("   📊 Univariate statistical selection...")
        selector_univariate = SelectKBest(score_func=f_classif, k='all')
        selector_univariate.fit(X, y)
        univariate_scores = selector_univariate.scores_
        feature_scores['univariate'] = dict(zip(X.columns, univariate_scores))
        
        # 2. Recursive Feature Elimination with Random Forest
        print("   🌳 Recursive feature elimination...")
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rfe = RFE(rf_selector, n_features_to_select=min(20, len(X.columns)))
        rfe.fit(X_scaled, y)
        rfe_ranking = rfe.ranking_
        feature_scores['rfe'] = dict(zip(X.columns, 1.0 / rfe_ranking))  # Invert ranking
        
        # 3. Tree-based feature importance
        print("   🏗️  Tree-based importance...")
        rf_importance = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        rf_importance.fit(X_scaled, y)
        tree_importance = rf_importance.feature_importances_
        feature_scores['tree_importance'] = dict(zip(X.columns, tree_importance))
        
        # 4. L1 regularization (Lasso)
        print("   🎯 L1 regularization importance...")
        from sklearn.feature_selection import SelectFromModel
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=self.random_state)
        lasso_selector = SelectFromModel(lasso, prefit=False)
        lasso_selector.fit(X_scaled, y)
        lasso_importance = np.abs(lasso_selector.estimator_.coef_[0])
        feature_scores['lasso'] = dict(zip(X.columns, lasso_importance))
        
        # 5. Mutual Information
        print("   🔗 Mutual information...")
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_scaled, y, random_state=self.random_state)
        feature_scores['mutual_info'] = dict(zip(X.columns, mi_scores))
        
        # Combine all scores using ensemble ranking
        print("   🎼 Combining feature selection methods...")
        combined_scores = self._combine_feature_scores(feature_scores)
        
        # Select top features
        n_features = min(30, len(X.columns))  # Select top 30 or all if less
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        selected_features = [feat for feat, score in top_features]
        
        # Create feature importance dataframe for analysis
        self.feature_importance_df = pd.DataFrame([
            {'feature': feat, 'combined_score': score, **{method: feature_scores[method].get(feat, 0) 
                                                         for method in feature_scores.keys()}}
            for feat, score in top_features
        ])
        
        print(f"   ✅ Selected {len(selected_features)} features from {len(X.columns)}")
        print(f"   🏆 Top 5 features: {selected_features[:5]}")
        
        return X[selected_features], selected_features
    
    def _combine_feature_scores(self, feature_scores):
        """Combine multiple feature selection scores using rank aggregation"""
        all_features = set()
        for scores in feature_scores.values():
            all_features.update(scores.keys())
        
        # Normalize scores to [0, 1] range for each method
        normalized_scores = {}
        for method, scores in feature_scores.items():
            max_score = max(scores.values())
            min_score = min(scores.values())
            if max_score > min_score:
                normalized_scores[method] = {
                    feat: (score - min_score) / (max_score - min_score)
                    for feat, score in scores.items()
                }
            else:
                normalized_scores[method] = {feat: 1.0 for feat in scores.keys()}
        
        # Average normalized scores
        combined_scores = {}
        for feature in all_features:
            scores = [normalized_scores[method].get(feature, 0) for method in normalized_scores.keys()]
            combined_scores[feature] = np.mean(scores)
        
        return combined_scores
    
    def prepare_data_with_sampling(self, X, y, test_size=0.2):
        """Prepare data with advanced sampling strategies"""
        print("\n🎲 DATA PREPARATION WITH SAMPLING")
        print("-" * 50)
        
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"   📊 Initial split - Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"   📈 Train positive rate: {y_train.mean():.1%}")
        print(f"   📉 Test positive rate: {y_test.mean():.1%}")
        
        # Apply multiple sampling strategies and evaluate
        sampling_strategies = {
            'SMOTE': SMOTE(random_state=self.random_state),
            'ADASYN': ADASYN(random_state=self.random_state),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=self.random_state),
            'SMOTEENN': SMOTEENN(random_state=self.random_state)
        }
        
        best_sampling = None
        best_score = 0
        sampling_results = {}
        
        print("   🧪 Evaluating sampling strategies...")
        
        for name, sampler in sampling_strategies.items():
            try:
                X_sampled, y_sampled = sampler.fit_resample(X_train, y_train)
                
                # Quick evaluation with RandomForest
                rf_eval = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                cv_scores = cross_val_score(rf_eval, X_sampled, y_sampled, 
                                          cv=3, scoring='roc_auc', n_jobs=-1)
                avg_score = cv_scores.mean()
                
                sampling_results[name] = {
                    'score': avg_score,
                    'size': len(X_sampled),
                    'positive_rate': y_sampled.mean()
                }
                
                print(f"      {name}: AUC={avg_score:.3f}, Size={len(X_sampled)}, Pos%={y_sampled.mean():.1%}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_sampling = name
                    
            except Exception as e:
                print(f"      {name}: Failed ({str(e)[:50]}...)")
        
        # Apply best sampling strategy
        if best_sampling:
            print(f"   🏆 Best sampling strategy: {best_sampling}")
            X_train_resampled, y_train_resampled = sampling_strategies[best_sampling].fit_resample(X_train, y_train)
        else:
            print("   ⚠️  Using original data (no sampling)")
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Scale features
        print("   ⚖️  Scaling features...")
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   ✅ Final training set: {len(X_train_resampled)} samples")
        print(f"   ✅ Positive rate after sampling: {y_train_resampled.mean():.1%}")
        
        return X_train_scaled, X_test_scaled, y_train_resampled, y_test, X_train_resampled, X_test
    
    def create_advanced_base_models(self):
        """Create a diverse set of base models with different strengths"""
        print("\n🤖 CREATING ADVANCED BASE MODELS")
        print("-" * 50)
        
        self.models = {
            # Tree-based models
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, 
                class_weight='balanced', n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, 
                random_state=self.random_state, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1, verbose=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=200, depth=8, learning_rate=0.1,
                random_seed=self.random_state, verbose=False
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8, 
                random_state=self.random_state
            ),
            
            # Linear models
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state, max_iter=2000, 
                class_weight='balanced', C=1.0
            ),
            'LogisticRegressionL1': LogisticRegression(
                random_state=self.random_state, max_iter=2000,
                class_weight='balanced', penalty='l1', solver='liblinear'
            ),
            
            # Other algorithms
            'SVM': SVC(
                probability=True, random_state=self.random_state,
                class_weight='balanced', kernel='rbf'
            ),
            'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'NaiveBayes': GaussianNB()
        }
        
        print(f"   ✅ Created {len(self.models)} base models:")
        for name in self.models.keys():
            print(f"      🔸 {name}")
    
    def hyperparameter_optimization_with_optuna(self, X_train, y_train, n_trials=100):
        """Advanced hyperparameter optimization using Optuna"""
        print(f"\n🎯 HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
        print("-" * 50)
        
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 8, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
            return cv_scores.mean()
        
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
            return cv_scores.mean()
        
        # Optimize RandomForest
        print("   🌳 Optimizing RandomForest...")
        study_rf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study_rf.optimize(objective_rf, n_trials=n_trials//2, show_progress_bar=False)
        
        best_rf_params = study_rf.best_params
        best_rf_params.update({'class_weight': 'balanced', 'random_state': self.random_state, 'n_jobs': -1})
        self.models['RandomForest'] = RandomForestClassifier(**best_rf_params)
        print(f"      Best RF AUC: {study_rf.best_value:.4f}")
        
        # Optimize XGBoost
        print("   🚀 Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study_xgb.optimize(objective_xgb, n_trials=n_trials//2, show_progress_bar=False)
        
        best_xgb_params = study_xgb.best_params
        best_xgb_params.update({'random_state': self.random_state, 'n_jobs': -1})
        self.models['XGBoost'] = xgb.XGBClassifier(**best_xgb_params)
        print(f"      Best XGB AUC: {study_xgb.best_value:.4f}")
        
        print("   ✅ Hyperparameter optimization completed")
    
    def comprehensive_model_evaluation(self, X_train, y_train, cv_folds=5):
        """Comprehensive evaluation of all models with detailed metrics"""
        print(f"\n📊 COMPREHENSIVE MODEL EVALUATION ({cv_folds}-fold CV)")
        print("-" * 50)
        
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        results = {}
        detailed_results = []
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for name, model in self.models.items():
            print(f"   🔄 Evaluating {name}...")
            
            model_results = {}
            for metric in metrics:
                scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, 
                                       scoring=metric, n_jobs=-1)
                model_results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
            
            results[name] = model_results
            
            # Store for summary table
            detailed_results.append({
                'Model': name,
                'Accuracy': f"{model_results['accuracy']['mean']:.3f} ± {model_results['accuracy']['std']:.3f}",
                'Precision': f"{model_results['precision']['mean']:.3f} ± {model_results['precision']['std']:.3f}",
                'Recall': f"{model_results['recall']['mean']:.3f} ± {model_results['recall']['std']:.3f}",
                'F1-Score': f"{model_results['f1']['mean']:.3f} ± {model_results['f1']['std']:.3f}",
                'ROC-AUC': f"{model_results['roc_auc']['mean']:.3f} ± {model_results['roc_auc']['std']:.3f}"
            })
            
            print(f"      ROC-AUC: {model_results['roc_auc']['mean']:.3f} ± {model_results['roc_auc']['std']:.3f}")
            print(f"      Recall:  {model_results['recall']['mean']:.3f} ± {model_results['recall']['std']:.3f}")
        
        # Create summary table
        results_df = pd.DataFrame(detailed_results)
        print(f"\n📋 CROSS-VALIDATION RESULTS SUMMARY:")
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(f"{self.results_dir}/cv_results.csv", index=False)
        
        self.cv_results = results
        return results
    
    def create_stacked_ensemble(self, X_train, y_train):
        """Create advanced stacked ensemble model"""
        print("\n🏗️  CREATING STACKED ENSEMBLE")
        print("-" * 50)
        
        # Select best performing models based on CV results
        if not self.cv_results:
            raise ValueError("Run comprehensive_model_evaluation first")
        
        # Rank models by ROC-AUC
        model_rankings = []
        for name, results in self.cv_results.items():
            auc_score = results['roc_auc']['mean']
            model_rankings.append((name, auc_score))
        
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        print("   📊 Model rankings by ROC-AUC:")
        for i, (name, score) in enumerate(model_rankings[:10], 1):
            print(f"      {i:2}. {name}: {score:.4f}")
        
        # Select top performers for ensemble
        top_models = model_rankings[:6]  # Top 6 models
        
        print(f"\n   🎯 Selected {len(top_models)} models for ensemble:")
        ensemble_estimators = []
        for name, score in top_models:
            ensemble_estimators.append((name.lower(), self.models[name]))
            print(f"      ✅ {name} (AUC: {score:.4f})")
        
        # Create stacked ensemble with meta-learner
        meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        self.ensemble_model = StackingClassifier(
            estimators=ensemble_estimators,
            final_estimator=meta_learner,
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print("   🔧 Training stacked ensemble...")
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        cv_score = cross_val_score(
            self.ensemble_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc', n_jobs=-1
        )
        
        print(f"   🏆 Ensemble CV ROC-AUC: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
        
        self.is_trained = True
        return cv_score
    
    def calibrate_probabilities(self, X_train, y_train):
        """Apply probability calibration to improve reliability"""
        print("\n📐 PROBABILITY CALIBRATION")
        print("-" * 50)
        
        if not self.is_trained:
            raise ValueError("Train ensemble model first")
        
        # Test different calibration methods
        calibration_methods = ['sigmoid', 'isotonic']
        calibration_results = {}
        
        for method in calibration_methods:
            print(f"   🔄 Testing {method} calibration...")
            
            calibrated = CalibratedClassifierCV(
                self.ensemble_model, method=method, cv=3, n_jobs=-1
            )
            
            # Evaluate calibration using cross-validation
            cv_scores = cross_val_score(calibrated, X_train, y_train, cv=3, scoring='neg_brier_score')
            brier_score = -cv_scores.mean()
            
            calibration_results[method] = {
                'brier_score': brier_score,
                'brier_std': cv_scores.std(),
                'model': calibrated
            }
            
            print(f"      Brier Score: {brier_score:.4f} ± {cv_scores.std():.4f}")
        
        # Select best calibration method
        best_method = min(calibration_results.keys(), 
                         key=lambda x: calibration_results[x]['brier_score'])
        
        print(f"   🏆 Best calibration method: {best_method}")
        
        # Train final calibrated model
        self.calibrated_ensemble = CalibratedClassifierCV(
            self.ensemble_model, method=best_method, cv=3, n_jobs=-1
        )
        self.calibrated_ensemble.fit(X_train, y_train)
        
        print("   ✅ Probability calibration completed")
        
        return calibration_results
    
    def find_optimal_threshold_advanced(self, X_test, y_test):
        """Advanced threshold optimization with multiple criteria"""
        print(f"\n🎯 ADVANCED THRESHOLD OPTIMIZATION")
        print("-" * 50)
        
        # Get probability predictions
        y_prob = self.calibrated_ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        # Calculate various metrics for each threshold
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:  # Avoid degenerate cases
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Calculate additional metrics
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Youden's J statistic (sensitivity + specificity - 1)
                youden_j = recall + specificity - 1
                
                threshold_results.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'f1': f1,
                    'youden_j': youden_j
                })
        
        results_df = pd.DataFrame(threshold_results)
        
        # Find optimal thresholds using different criteria
        optimal_thresholds = {}
        
        # 1. Target sensitivity threshold
        target_sensitivity_results = results_df[results_df['recall'] >= self.target_sensitivity]
        if not target_sensitivity_results.empty:
            best_idx = target_sensitivity_results['specificity'].idxmax()
            optimal_thresholds['target_sensitivity'] = results_df.loc[best_idx]
        
        # 2. Youden's J statistic (max sensitivity + specificity - 1)
        best_youden_idx = results_df['youden_j'].idxmax()
        optimal_thresholds['youden_j'] = results_df.loc[best_youden_idx]
        
        # 3. F1-score maximization
        best_f1_idx = results_df['f1'].idxmax()
        optimal_thresholds['f1_max'] = results_df.loc[best_f1_idx]
        
        # 4. Balanced accuracy (average of sensitivity and specificity)
        results_df['balanced_accuracy'] = (results_df['recall'] + results_df['specificity']) / 2
        best_balanced_idx = results_df['balanced_accuracy'].idxmax()
        optimal_thresholds['balanced'] = results_df.loc[best_balanced_idx]
        
        # Display results
        print("   📊 OPTIMAL THRESHOLDS BY DIFFERENT CRITERIA:")
        print("-" * 50)
        
        for criterion, result in optimal_thresholds.items():
            print(f"   🎯 {criterion.upper()}:")
            print(f"      Threshold: {result['threshold']:.4f}")
            print(f"      Sensitivity: {result['recall']:.3f}")
            print(f"      Specificity: {result['specificity']:.3f}")
            print(f"      F1-Score: {result['f1']:.3f}")
            print(f"      Accuracy: {result['accuracy']:.3f}")
            print()
        
        # Select threshold based on target sensitivity if available, otherwise Youden's J
        if 'target_sensitivity' in optimal_thresholds:
            selected_criterion = 'target_sensitivity'
            self.optimal_threshold = optimal_thresholds['target_sensitivity']['threshold']
            print(f"   ✅ Selected: TARGET SENSITIVITY threshold = {self.optimal_threshold:.4f}")
        else:
            selected_criterion = 'youden_j'
            self.optimal_threshold = optimal_thresholds['youden_j']['threshold']
            print(f"   ✅ Selected: YOUDEN'S J threshold = {self.optimal_threshold:.4f}")
        
        # Save threshold analysis
        results_df.to_csv(f"{self.results_dir}/threshold_analysis.csv", index=False)
        
        return optimal_thresholds, selected_criterion
    
    def comprehensive_final_evaluation(self, X_test, y_test):
        """Comprehensive evaluation on test set with visualizations"""
        print("\n📈 COMPREHENSIVE FINAL EVALUATION")
        print("=" * 50)
        
        # Get predictions
        y_prob = self.calibrated_ensemble.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.optimal_threshold).astype(int)
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)
        brier_score = brier_score_loss(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        # Print detailed results
        print(f"📊 FINAL TEST SET RESULTS:")
        print(f"   🎯 Accuracy:           {accuracy:.4f}")
        print(f"   🎯 Precision:          {precision:.4f}")
        print(f"   🎯 Recall (Sensitivity): {recall:.4f}")
        print(f"   🎯 Specificity:        {specificity:.4f}")
        print(f"   🎯 F1-Score:           {f1:.4f}")
        print(f"   🎯 ROC-AUC:           {roc_auc:.4f}")
        print(f"   🎯 Average Precision:  {average_precision:.4f}")
        print(f"   🎯 Brier Score:        {brier_score:.4f}")
        print(f"   🎯 Matthews Corr Coef: {mcc:.4f}")
        print()
        print(f"📋 CONFUSION MATRIX:")
        print(f"   True Negatives:  {tn}")
        print(f"   False Positives: {fp}")
        print(f"   False Negatives: {fn}")
        print(f"   True Positives:  {tp}")
        
        # Store results
        self.results = {
            'test_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'average_precision': average_precision,
                'brier_score': brier_score,
                'matthews_corrcoef': mcc
            },
            'confusion_matrix': cm,
            'optimal_threshold': self.optimal_threshold,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        # Create visualizations
        self._create_evaluation_plots(y_test, y_prob, y_pred)
        
        return self.results
    
    def _create_evaluation_plots(self, y_test, y_prob, y_pred):
        """Create comprehensive evaluation plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        axes[0, 1].plot(recall, precision, color='darkgreen', lw=2, label=f'PR (AP = {avg_precision:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
        axes[0, 2].set_title('Confusion Matrix')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # 4. Probability Distribution
        axes[1, 0].hist(y_prob[y_test == 0], bins=30, alpha=0.7, label='Negative Class', color='lightcoral')
        axes[1, 0].hist(y_prob[y_test == 1], bins=30, alpha=0.7, label='Positive Class', color='lightblue')
        axes[1, 0].axvline(self.optimal_threshold, color='red', linestyle='--', label=f'Threshold = {self.optimal_threshold:.3f}')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Calibration Plot
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        axes[1, 1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        axes[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Importance (Top 10)
        if self.feature_importance_df is not None:
            top_features = self.feature_importance_df.head(10)
            axes[1, 2].barh(range(len(top_features)), top_features['combined_score'])
            axes[1, 2].set_yticks(range(len(top_features)))
            axes[1, 2].set_yticklabels(top_features['feature'])
            axes[1, 2].set_xlabel('Importance Score')
            axes[1, 2].set_title('Top 10 Feature Importance')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/evaluation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Evaluation plots saved to: {self.results_dir}/evaluation_plots.png")
    
    def save_complete_model(self, filepath=None):
        """Save the complete trained model with all artifacts"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if filepath is None:
            filepath = f"models/advanced_heart_risk_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'calibrated_ensemble': self.calibrated_ensemble,
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'feature_importance': self.feature_importance_df,
            'training_results': self.results,
            'cv_results': self.cv_results,
            'target_sensitivity': self.target_sensitivity,
            'model_version': 'AdvancedHeartRiskPredictor_v1.0',
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        
        print(f"✅ Complete model saved to: {filepath}")
        print(f"   📦 Model size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        
        # Also save a summary report
        self._save_training_report()
        
        return filepath
    
    def _save_training_report(self):
        """Save a comprehensive training report"""
        report_path = f"{self.results_dir}/training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("ADVANCED HEART RISK PREDICTOR - TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Version: AdvancedHeartRiskPredictor_v1.0\n")
            f.write(f"Random State: {self.random_state}\n\n")
            
            if self.results:
                f.write("FINAL TEST RESULTS:\n")
                f.write("-" * 30 + "\n")
                for metric, value in self.results['test_metrics'].items():
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
                f.write(f"\nOptimal Threshold: {self.optimal_threshold:.4f}\n")
                f.write(f"Target Sensitivity: {self.target_sensitivity:.3f}\n\n")
            
            if self.feature_importance_df is not None:
                f.write("TOP 15 FEATURES:\n")
                f.write("-" * 30 + "\n")
                for idx, row in self.feature_importance_df.head(15).iterrows():
                    f.write(f"{row['feature']}: {row['combined_score']:.4f}\n")
        
        print(f"   📄 Training report saved to: {report_path}")
    
    def run_complete_training_pipeline(self, data_filepath='data/processed/heart_disease_engineered.csv'):
        """Execute the complete advanced training pipeline"""
        print("🚀 STARTING ADVANCED HEART RISK PREDICTOR TRAINING")
        print("=" * 80)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. Load and validate data
            X, y = self.load_and_validate_data(data_filepath)
            if X is None:
                return None
            
            # 2. Advanced feature engineering
            X_engineered = self.advanced_feature_engineering(X, y)
            
            # 3. Intelligent feature selection
            X_selected, selected_features = self.intelligent_feature_selection(X_engineered, y)
            
            # 4. Data preparation with sampling
            X_train, X_test, y_train, y_test, _, _ = self.prepare_data_with_sampling(X_selected, y)
            
            # 5. Create base models
            self.create_advanced_base_models()
            
            # 6. Hyperparameter optimization
            self.hyperparameter_optimization_with_optuna(X_train, y_train, n_trials=50)
            
            # 7. Comprehensive evaluation
            cv_results = self.comprehensive_model_evaluation(X_train, y_train)
            
            # 8. Create stacked ensemble
            ensemble_cv_score = self.create_stacked_ensemble(X_train, y_train)
            
            # 9. Probability calibration
            calibration_results = self.calibrate_probabilities(X_train, y_train)
            
            # 10. Threshold optimization
            threshold_results, selected_criterion = self.find_optimal_threshold_advanced(X_test, y_test)
            
            # 11. Final evaluation
            final_results = self.comprehensive_final_evaluation(X_test, y_test)
            
            # 12. Save model and artifacts
            model_path = self.save_complete_model()
            
            # Training summary
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            print("\n" + "=" * 80)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"⏱️  Training Duration: {training_duration}")
            print(f"🎯 Final ROC-AUC: {final_results['test_metrics']['roc_auc']:.4f}")
            print(f"🎯 Final Sensitivity: {final_results['test_metrics']['recall']:.4f}")
            print(f"🎯 Final Specificity: {final_results['test_metrics']['specificity']:.4f}")
            print(f"🎯 Optimal Threshold: {self.optimal_threshold:.4f}")
            print(f"📁 Results Directory: {self.results_dir}")
            print(f"💾 Model Saved: {model_path}")
            print("=" * 80)
            
            return {
                'model_path': model_path,
                'results_dir': self.results_dir,
                'final_results': final_results,
                'training_duration': training_duration
            }
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {str(e)}")
            raise e

# Main execution
if __name__ == "__main__":
    # Initialize and run training
    predictor = AdvancedHeartRiskPredictor(random_state=42, verbose=True)
    
    # Run complete training pipeline
    training_results = predictor.run_complete_training_pipeline()
    
    if training_results:
        print(f"\n🎯 Training completed! Check results in: {training_results['results_dir']}")
    else:
        print("\n❌ Training failed!")