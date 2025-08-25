import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import shap
from datetime import datetime
from typing import Dict, Optional

class GoalAchievementPredictor:
    """
    ML model to predict likelihood of customer achieving financial goals
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.feature_scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.shap_explainer = None
        if model_path:
            self.load_model(model_path)

    def engineer_features(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        features['age_normalized'] = customer_data['age'] / 100
        features['monthly_income_log'] = np.log1p(customer_data['monthly_income'])
        features['savings_rate'] = customer_data['savings_rate'].clip(0, 1)
        features['current_savings_log'] = np.log1p(customer_data['current_savings'])
        features['goal_amount_log'] = np.log1p(customer_data['goal_amount'])
        features['goal_timeline_months'] = customer_data['goal_timeline_months']
        features['feasibility_ratio'] = (
            (customer_data['monthly_income'] * customer_data['savings_rate'] * customer_data['goal_timeline_months']) / customer_data['goal_amount']
        )
        features['digital_engagement'] = customer_data.get('digital_engagement_score', 0.5)
        features['has_investment'] = customer_data.get('has_investment_account', False).astype(int)
        self.feature_names = list(features.columns)
        return features.fillna(features.median())

    def train_model(self, training_data: pd.DataFrame, target_column: str = 'goal_achieved') -> Dict:
        features = self.engineer_features(training_data)
        target = training_data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        features_scaled = self.feature_scaler.fit_transform(X_train)
        self.model.fit(features_scaled, y_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.is_trained = True
        return {
            'test_accuracy': test_accuracy,
            'feature_count': len(self.feature_names),
            'training_samples': len(training_data),
            'training_date': datetime.now().isoformat()
        }

    async def predict(self, customer_features: Dict) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        customer_df = pd.DataFrame([customer_features])
        features = self.engineer_features(customer_df)
        features_scaled = self.feature_scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0, 1]
        explanations = {}
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(features_scaled)
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                feature_importance = dict(zip(self.feature_names, shap_values[0]))
                top_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                explanations = {
                    'top_factors': [{'feature': f, 'impact': float(i)} for f, i in top_factors]
                }
            except Exception as e:
                explanations = {'error': f'SHAP explanation failed: {str(e)}'}
        return {
            'success_probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 2),
            'explanations': explanations,
            'prediction_timestamp': datetime.now().isoformat()
        }

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("No trained model to save")
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'shap_explainer': self.shap_explainer
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_names = model_data['feature_names']
            self.shap_explainer = model_data.get('shap_explainer')
            self.is_trained = True
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
