import joblib
import numpy as np
import pandas as pd
from .preprocess import create_clinical_features

class DiabetesPredictor:
    def __init__(self, model_path='models/diabetes_random_forest.pkl'):
        """
        Initialize diabetes predictor
        
        Args:
            model_path (str): Path to trained model file (.pkl)
        """
        # Load the trained machine learning model from disk
        self.model = joblib.load(model_path)
        
        # These are the base input features expected from the user/patient
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    
    def predict(self, features, return_prob=True):
        """
        Predict diabetes risk
        
        Args:
            features (list/dict): Patient data (either as list or dictionary)
            return_prob (bool): If True, return probability; else return class (0 or 1)
            
        Returns:
            float or int: Diabetes risk (probability or class label)
        """
        # Convert input into a pandas DataFrame (model expects tabular data)
        if isinstance(features, dict):
            # If user gives a dictionary: {'Glucose': 120, 'Age': 35, ...}
            input_data = pd.DataFrame([features])
        elif isinstance(features, list):
            # If user gives a list: [2, 120, 70, 20, 100, 26.2, 0.5, 35]
            input_data = pd.DataFrame([features], columns=self.feature_names)
        else:
            raise ValueError("Input must be list or dictionary")
        
        # Add extra engineered clinical features (e.g., BMI categories, ratios, etc.)
        # This function should return a DataFrame with additional columns
        input_data = create_clinical_features(input_data)
        
        # Ensure the column order matches what the model was trained on
        # This avoids wrong predictions due to column mismatch
        input_data = input_data[self.model.feature_names_in_]
        
        # Make prediction
        if return_prob:
            # predict_proba returns [prob_class_0, prob_class_1]
            # We take the probability of class 1 (diabetes)
            return self.model.predict_proba(input_data)[0][1]
        
        # Otherwise, return the predicted class (0 = no diabetes, 1 = diabetes)
        return self.model.predict(input_data)[0]
    
    def explain_prediction(self, features):
        """
        Generate explanation for prediction using SHAP values
        
        Requires: pip install shap
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return None
        
        # Prepare input data in the same way as for prediction
        input_data = pd.DataFrame([features], columns=self.feature_names)
        input_data = create_clinical_features(input_data)
        input_data = input_data[self.model.feature_names_in_]
        
        # Create a SHAP explainer for tree-based models (like Random Forest)
        explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values for this input
        shap_values = explainer.shap_values(input_data)
        
        # Return a simple dictionary with:
        # - base_value: average model output
        # - shap_values: contribution of each feature
        # - prediction: final predicted probability
        return {
            'base_value': explainer.expected_value[1],
            'shap_values': dict(zip(input_data.columns, shap_values[1][0])),
            'prediction': self.model.predict_proba(input_data)[0][1]
        }

# Example usage
if __name__ == "__main__":
    # Initialize the predictor object (loads the trained model)
    predictor = DiabetesPredictor()
    
    # Sample patient data (example input)
    patient = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 100,
        'BMI': 26.2,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 35
    }
    
    # Get prediction as probability
    risk = predictor.predict(patient)
    print(f"Diabetes risk: {risk:.1%}")
    
    # Get SHAP explanation for the prediction
    explanation = predictor.explain_prediction(patient)
    if explanation:
        print("\nPrediction Explanation:")
        for feature, value in explanation['shap_values'].items():
            print(f"{feature}: {value:.4f}")
        print(f"Base value: {explanation['base_value']:.4f}")
        print(f"Final prediction: {explanation['prediction']:.4f}")