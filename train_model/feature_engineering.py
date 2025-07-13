import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class HeartRiskFeatureEngineer:
    """Class to create advanced medical features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath='data/raw/cardio_train.csv'):
        """Load data and perform initial cleaning."""
        
        df = pd.read_csv(filepath, sep=';')
        print(f"Data loaded: {len(df)} records")
        
        # Handle missing values if they exist
        if df.isnull().sum().sum() > 0:
            print("Cleaning missing values...")
            # Impute missing values with the median for numeric variables
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Rename cardio to target
        df.rename(columns={'cardio': 'target'}, inplace=True)
        
        print("[OK] Data cleaned and target renamed")
        return df
    
    def create_age_features(self, df):
        """Create age-related features."""
        
        # Convert age from days to years
        df['age'] = df['age'] / 365.25

        # Medically relevant age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 45, 55, 65, 100], 
                                labels=['<45', '45-55', '55-65', '65+'])
        
        # Normalized age
        df['age_normalized'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        
        # Exponential age risk (increases exponentially after 45)
        df['age_risk_exponential'] = np.where(df['age'] > 45, 
                                            np.exp((df['age'] - 45) / 10), 
                                            1.0)
        
        print("[OK] Age features created")
        return df
    
    def create_cardiac_features(self, df):
        """Create advanced cardiac features."""
        
        # Blood pressure categories
        df['bp_category'] = pd.cut(df['ap_hi'], 
                                 bins=[0, 120, 140, 160, 180, 1000], 
                                 labels=['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis'])
        
        # Pulse pressure
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

        print("[OK] Cardiac features created")
        return df
    
    def create_metabolic_features(self, df):
        """Create metabolic features."""
        
        # Metabolic profile (cholesterol adjusted for age)
        df['metabolic_profile'] = df['cholesterol'] / df['age']
        
        # Cholesterol categories according to medical guidelines
        df['chol_category'] = pd.cut(df['cholesterol'], 
                                   bins=[0, 1, 2, 3, 1000], 
                                   labels=['Normal', 'Above Normal', 'Well Above Normal', 'High'])
        
        # Simplified metabolic syndrome (combination of factors)
        df['metabolic_syndrome_risk'] = (
            (df['cholesterol'] > 1).astype(int) + 
            (df['gluc'] > 1).astype(int) + 
            (df['ap_hi'] > 140).astype(int)
        )
        
        print("[OK] Metabolic features created")
        return df
    
    
    
    
    
    def create_gender_interaction_features(self, df):
        """Create gender interaction features."""
        
        # Women and men have different risk patterns
        df['male_age_interaction'] = df['gender'] * df['age']
        df['female_chol_interaction'] = (1 - df['gender']) * df['cholesterol']
        
        # Gender-specific risk
        df['gender_specific_risk'] = np.where(
            df['gender'] == 2,  # Men
            df['age'] * 0.1 + df['cholesterol'] * 0.005,  # Greater weight on age
            df['cholesterol'] * 0.008  # Women: greater weight on cholesterol
        )
        
        print("[OK] Gender interaction features created")
        return df
    
    def create_composite_risk_scores(self, df):
        """Create composite risk scores."""
        
        # Traditional risk score (based on simplified Framingham)
        df['traditional_risk_score'] = (
            df['age'] * 0.04 + 
            df['gender'] * 10 +  # Men have a higher baseline risk
            (df['cholesterol'] - 1) * 20 + 
            df['ap_hi'] * 0.1 + 
            df['gluc'] * 20
        )
        
        # Cardiac risk score (heart-specific factors)
        df['cardiac_risk_score'] = (
            df['pulse_pressure'] * 0.2 + 
            df['ap_hi'] * 0.1
        )
        
        # Normalized combined score
        df['combined_risk_score'] = (
            df['traditional_risk_score'] * 0.4 + 
            df['cardiac_risk_score'] * 0.6
        )
        
        print("[OK] Composite risk scores created")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables."""
        
        # Categorical variables for encoding
        categorical_cols = ['age_group', 'chol_category', 'bp_category']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        print("[OK] Categorical variables encoded")
        return df
    
    def select_best_features(self, df, target_col='target', k=20):
        """Select the best features."""
        
        # Separate numeric features
        feature_cols = [col for col in df.columns if col != target_col and 
                       df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Select k best features
        selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        # Get names of selected features
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        scores = selector.scores_[selector.get_support()]
        
        # Create DataFrame with scores
        feature_scores = pd.DataFrame({
            'feature': selected_features,
            'score': scores
        }).sort_values('score', ascending=False)
        
        print(f"[OK] {k} best features selected")
        print("\nTop 10 most important features:")
        for i, (_, row) in enumerate(feature_scores.head(10).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['score']:.2f}")
        
        return selected_features, feature_scores
    
    def process_complete_dataset(self, filepath='data/raw/cardio_train.csv'):
        """Process the complete dataset with all features."""
        
        print("Starting Complete Feature Engineering...")
        print("=" * 60)
        
        # Load data
        df = self.load_data(filepath)
        
        # Create all features
        df = self.create_age_features(df)
        df = self.create_cardiac_features(df) 
        df = self.create_metabolic_features(df)
        df = self.create_gender_interaction_features(df)
        df = self.create_composite_risk_scores(df)
        df = self.encode_categorical_features(df)
        
        # Select best features
        selected_features, feature_scores = self.select_best_features(df)
        
        # Create final dataset with selected features + target
        final_features = selected_features + ['target']
        df_final = df[final_features]
        
        # Save processed dataset
        df_final.to_csv('data/processed/heart_disease_engineered.csv', index=False)
        feature_scores.to_csv('data/processed/feature_importance_scores.csv', index=False)
        
        print(f"\n[OK] Feature Engineering completed!")
        print(f"Final dataset: {len(df_final)} rows, {len(df_final.columns)-1} features")
        print(f"Saved to: data/processed/heart_disease_engineered.csv")
        
        return df_final, feature_scores

def create_feature_visualization(df, feature_scores):
    """Create visualizations of the new features."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Feature Engineering - New Medical Features', fontsize=16, fontweight='bold')
    
    # 1. Top 15 most important features
    top_features = feature_scores.head(15)
    axes[0,0].barh(range(len(top_features)), top_features['score'])
    axes[0,0].set_yticks(range(len(top_features)))
    axes[0,0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,0].set_title('Top 15 Most Important Features')
    axes[0,0].set_xlabel('F-statistic Score')
    
    # 2. Distribution of the combined risk score
    if 'combined_risk_score' in df.columns:
        axes[0,1].hist(df[df['target']==0]['combined_risk_score'], alpha=0.7, 
                      label='No disease', bins=20, color='lightblue')
        axes[0,1].hist(df[df['target']==1]['combined_risk_score'], alpha=0.7, 
                      label='With disease', bins=20, color='lightcoral')
        axes[0,1].set_title('Combined Risk Score')
        axes[0,1].set_xlabel('Score')
        axes[0,1].legend()
    
    # 3. Cardiac capacity by age group
    if 'cardiac_capacity_index' in df.columns and 'age_group' in df.columns:
        df.boxplot(column='cardiac_capacity_index', by='target', ax=axes[0,2])
        axes[0,2].set_title('Cardiac Capacity Index')
        axes[0,2].set_xlabel('Heart Disease')
        axes[0,2].set_ylabel('Capacity Index')
    
    # 4. Metabolic profile
    if 'metabolic_profile' in df.columns:
        axes[1,0].scatter(df['age'], df['metabolic_profile'], 
                         c=df['target'], cmap='coolwarm', alpha=0.6)
        axes[1,0].set_title('Metabolic Profile vs Age')
        axes[1,0].set_xlabel('Age')
        axes[1,0].set_ylabel('Metabolic Profile')
    
    # 5. Risk by metabolic syndrome
    if 'metabolic_syndrome_risk' in df.columns:
        metabolic_risk = pd.crosstab(df['metabolic_syndrome_risk'], df['target'])
        metabolic_risk.plot(kind='bar', ax=axes[1,1], color=['lightblue', 'lightcoral'])
        axes[1,1].set_title('Risk by Metabolic Syndrome')
        axes[1,1].set_xlabel('Metabolic Risk Factors')
        axes[1,1].legend(['No disease', 'With disease'])
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=0)
    
    # 6. Correlation between new features and target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 10:
        correlations = df[numeric_cols].corr()['target'].abs().sort_values(ascending=False)[1:16]
        axes[1,2].barh(range(len(correlations)), correlations.values)
        axes[1,2].set_yticks(range(len(correlations)))
        axes[1,2].set_yticklabels(correlations.index, fontsize=8)
        axes[1,2].set_title('Correlation with Target (Top 15)')
        axes[1,2].set_xlabel('Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig('reports/feature_engineering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to reports/feature_engineering_analysis.png")

def generate_feature_insights(df, feature_scores):
    """Generate insights from the new features."""
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING INSIGHTS")
    print("="*60)
    
    # Top 5 most predictive features
    top_5 = feature_scores.head(5)
    print(f"\nTop 5 most predictive features:")
    for i, (_, row) in enumerate(top_5.iterrows()):
        print(f"   {i+1}. {row['feature']}: {row['score']:.2f}")
    
    # Analysis of the combined score
    if 'combined_risk_score' in df.columns:
        low_risk = df[df['target']==0]['combined_risk_score'].mean()
        high_risk = df[df['target']==1]['combined_risk_score'].mean()
        print(f"\nCombined risk score:")
        print(f"   No disease: {low_risk:.2f}")
        print(f"   With disease: {high_risk:.2f}")
        print(f"   Difference: {high_risk-low_risk:.2f} ({(high_risk-low_risk)/low_risk*100:.1f}% higher)")
    
    # Analysis of cardiac capacity
    if 'cardiac_capacity_index' in df.columns:
        cardiac_healthy = df[df['target']==0]['cardiac_capacity_index'].mean()
        cardiac_disease = df[df['target']==1]['cardiac_capacity_index'].mean()
        print(f"\nCardiac capacity index:")
        print(f"   No disease: {cardiac_healthy:.2f}")
        print(f"   With disease: {cardiac_disease:.2f}")
        print(f"   Difference: {cardiac_healthy-cardiac_disease:.2f} ({(cardiac_healthy-cardiac_disease)/cardiac_disease*100:.1f}% higher in healthy individuals)")
    
    # Metabolic syndrome
    if 'metabolic_syndrome_risk' in df.columns:
        metabolic_analysis = df.groupby('metabolic_syndrome_risk')['target'].agg(['count', 'mean'])
        print(f"\nMetabolic syndrome analysis:")
        for risk_level, data in metabolic_analysis.iterrows():
            print(f"   {risk_level} factors: {data['mean']:.1%} risk ({data['count']} cases)")
    
    # Improvement in class separation
    original_corr = abs(df['age'].corr(df['target']))
    if len(feature_scores) > 0:
        best_corr = feature_scores.iloc[0]['score'] / 100  # Normalize for comparison
        print(f"\nImprovement in predictive power:")
        print(f"   Age (original): {original_corr:.3f}")
        print(f"   Best feature: {best_corr:.3f}")
        if best_corr > original_corr:
            print(f"   Improvement: {(best_corr-original_corr)/original_corr*100:.1f}%")

# Execute complete Feature Engineering
if __name__ == "__main__":
    print("Starting Advanced Medical Feature Engineering...")
    print("=" * 70)
    
    # Initialize feature engineer
    engineer = HeartRiskFeatureEngineer()
    
    # Process complete dataset
    df_engineered, feature_scores = engineer.process_complete_dataset()
    
    # Create visualizations
    create_feature_visualization(df_engineered, feature_scores)
    
    # Generate insights
    generate_feature_insights(df_engineered, feature_scores)
    
    print(f"\nFEATURE ENGINEERING SUMMARY:")
    print(f"   {len(df_engineered)} records processed")
    print(f"   {len(df_engineered.columns)-1} final features")
    print(f"   Top feature: {feature_scores.iloc[0]['feature']}")
    print(f"   Dataset saved to data/processed/")
    print(f"\n[OK] Feature Engineering completed successfully!")
    print(f"[OK] Ready for model training!")