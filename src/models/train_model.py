import yaml
import logging
import argparse
import os
import pickle
import pandas as pd
import warnings
# Ignore feature name warnings from LightGBM/Sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier


# Logging config
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config_path):
    params = load_params(config_path)
    
    # --- 1. Path Setup ---
    base_dir = params['data_ingestion']['processed_dir']
    
    # Input: Training Features (X and y)
    input_train_feat = os.path.join(base_dir, params['build_features']['output_train_features'])
    
    # Output: Trained Model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    output_model_path = os.path.join(model_dir, "stacking_model.pkl")

    try:
        # --- 2. Load Data ---
        logging.info(f"Loading training data from {input_train_feat}...")
        with open(input_train_feat, 'rb') as f:
            train_data = pickle.load(f)
        
        X_train = train_data['X']
        y_train = train_data['y']

        # Ensure y is proper format (integers)
        y_train = y_train.astype(int)
        
        logging.info(f"Training Data Shape: {X_train.shape}")

        # --- 3. Initialize Base Models ---
        logging.info("Initializing base learners (LightGBM & LogReg)...")
        
        # Load parameters from yaml
        lgbm_config = params['model_params']['base_model_lgbm']
        logreg_config = params['model_params']['base_model_logreg']
        knn_config = params['model_params']['meta_model_knn']
        stacking_config = params['model_params']['stacking_config']

        # Define Base Learners
        # Note: We use **lgbm_config to unpack the dictionary into arguments
        lgbm = LGBMClassifier(**lgbm_config, random_state=42, verbose=-1)
        logreg = LogisticRegression(**logreg_config, random_state=42)

        estimators = [
            ('lgbm', lgbm),
            ('logreg', logreg)
        ]

        # --- 4. Initialize Meta Learner ---
        logging.info("Initializing meta learner (KNN)...")
        knn = KNeighborsClassifier(**knn_config)

        # --- 5. Build Stacking Ensemble ---
        logging.info("Building StackingClassifier...")
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=knn,
            cv=stacking_config['cv'],
            n_jobs=stacking_config['n_jobs'],
            passthrough=False # Meta-learner only sees predictions of base models
        )

        # --- 6. Train ---
        logging.info("Fitting the model... (This may take a while)")
        stacking_model.fit(X_train, y_train)
        
        # --- 7. Save Model ---
        logging.info(f"Saving trained model to {output_model_path}...")
        with open(output_model_path, 'wb') as f:
            pickle.dump(stacking_model, f)

        logging.info("Model training completed successfully.")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    train_model(config_path=args.config)