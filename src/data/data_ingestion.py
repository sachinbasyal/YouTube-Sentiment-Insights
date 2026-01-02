import os
import yaml
import argparse
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(config_path):
    """Load parameters from the YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ingest_data(config_path):
    params = load_params(config_path)
    
    # Extract ingestion config
    ingest_config = params['data_ingestion']
    source_path = ingest_config['source_path']
    processed_dir = ingest_config['processed_dir']
    test_size = ingest_config['test_size']
    random_state = ingest_config['random_state']
    mapping_dict = ingest_config['target_mapping']

    logging.info(f"Loading data from {source_path}...")

    try:
        # Load Data
        df = pd.read_csv(source_path)
        
        # Basic Cleaning: Drop nulls in text or category
        # Assuming columns are 'clean_comment' and 'category' based on standard Reddit_Data.csv
        # Adjust column names if your CSV differs.
        df.columns = ['clean_comment', 'category'] 
        initial_count = len(df)
        df.dropna(inplace=True)
        logging.info(f"Dropped {initial_count - len(df)} rows containing null values.")

        # --- CRITICAL MAPPING APPLICATION ---
        # Remapping -1 to 2 to ensure classes are [0, 1, 2] for LGBM/XGBoost
        logging.info(f"Applying target mapping: {mapping_dict}")
        
        # Ensure category is integer type before mapping
        df['category'] = df['category'].astype(int)
        df['category'] = df['category'].map(mapping_dict)

        # Check for unmapped values
        if df['category'].isnull().any():
            raise ValueError("Found target values that were not covered by the mapping dictionary.")

        logging.info(f"Class distribution after mapping:\n{df['category'].value_counts()}")

        # Split Data (Stratified to maintain class balance)
        logging.info("Splitting data into Train and Test sets...")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['category']
        )

        # Create processed directory
        os.makedirs(processed_dir, exist_ok=True)

        # Save Artifacts
        train_path = os.path.join(processed_dir, "train.csv")
        test_path = os.path.join(processed_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Ingestion completed. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        logging.info(f"Data saved to {processed_dir}")

    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    ingest_data(config_path=args.config)