import sys
import os
import yaml
import logging
import argparse
import pickle
import numpy as np

import warnings
# Ignore feature name warnings from LightGBM/Sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

# Add the 'src' directory to Python path so we can import modules from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the exact cleaning function used in training
# This guarantees Consistency (Training-Serving Skew prevention)—the exact same cleaning logic used for training is used for prediction.
from src.data.data_preprocessing import clean_text

# Logging config
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ModelPredictor:
    def __init__(self, config_path="params.yaml"):
        self.params = load_params(config_path)
        self.base_dir = self.params['data_ingestion']['processed_dir']
        self.model_dir = "models"
        
        # Load Artifacts immediately
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the Vectorizer and the Stacking Model."""
        vectorizer_path = os.path.join(self.base_dir, self.params['build_features']['output_vectorizer'])
        model_path = os.path.join(self.model_dir, "stacking_model.pkl")

        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or Vectorizer not found. Did you run 'dvc repro'?")

        logging.info("Loading vectorizer and model...")
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, text):
        """
        End-to-end prediction pipeline for a single string.
        Raw Text -> Clean -> Vectorize -> Predict -> Label
        """
        # 1. Preprocess (Clean)
        # We read the config to ensure we use same settings as training
        prep_config = self.params['preprocessing']
        cleaned_text = clean_text(
            text, 
            remove_html=prep_config['remove_html'], 
            remove_urls=prep_config['remove_urls'],
            lower=prep_config['lower_case']
        )

        # 2. Vectorize
        # Transform expects a list/iterable, so we wrap the string in a list
        text_vectorized = self.vectorizer.transform([cleaned_text])

        # 3. Predict
        prediction = self.model.predict(text_vectorized)[0]
        
        # 4. Map back to Readable Label
        # Your params.yaml mapping: {-1: 2, 0: 0, 1: 1}
        # So: 2 -> Negative, 0 -> Neutral, 1 -> Positive
        label_map = {
            2: "Negative",
            0: "Neutral",
            1: "Positive"
        }
        
        sentiment = label_map.get(int(prediction), "Unknown")
        return sentiment, cleaned_text

if __name__ == "__main__":
    # CLI Usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to classify", required=False)
    args = parser.parse_args()

    predictor = ModelPredictor()

    if args.text:
        # One-off prediction via command line
        sentiment, clean_version = predictor.predict(args.text)
        print(f"\n--- Result ---")
        print(f"Input: {args.text}")
        print(f"Cleaned: {clean_version}")
        print(f"Prediction: {sentiment}")
        print(f"--------------\n")
    else:
        # Interactive Mode
        print("\n=== YouTube Sentiment Predictor (Type 'exit' to quit) ===")
        while True:
            user_input = input("Enter comment: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            try:
                sentiment, _ = predictor.predict(user_input)
                print(f">> Sentiment: {sentiment}\n")
            except Exception as e:
                print(f"Error: {e}")