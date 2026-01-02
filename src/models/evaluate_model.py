import yaml
import logging
import argparse
import os
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Logging config
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(config_path):
    params = load_params(config_path)
    
    # --- 1. Path Setup ---
    base_dir = params['data_ingestion']['processed_dir']
    model_dir = "models"
    
    # Inputs
    input_test_feat = os.path.join(base_dir, params['build_features']['output_test_features'])
    model_path = os.path.join(model_dir, "stacking_model.pkl")
    
    # Outputs
    metrics_path = params['evaluation']['metrics_file']

    try:
        # --- 2. Load Data & Model ---
        logging.info("Loading test data and model...")
        
        with open(input_test_feat, 'rb') as f:
            test_data = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        X_test = test_data['X']
        y_test = test_data['y'].astype(int)

        # --- 3. Generate Predictions ---
        logging.info("Predicting on test set...")
        y_pred = model.predict(X_test)

        # --- 4. Calculate Metrics ---
        # We use 'weighted' average because classes might be imbalanced
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # --- 5. Save Metrics for DVC ---
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Metrics saved to {metrics_path}")

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    evaluate_model(config_path=args.config)