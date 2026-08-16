import os
import urllib.request
import pandas as pd
from src.utils import setup_logging

logger = setup_logging()

# Raw GitHub source URL for the dataset
DATA_URL = "https://raw.githubusercontent.com/ahmedgalalxxx/Walmart-Sales-Prediction/master/Walmart.csv"
LOCAL_DIR = "data"
LOCAL_FILE = os.path.join(LOCAL_DIR, "walmart.csv")

def download_walmart_data() -> str:
    """
    Downloads the Walmart Sales dataset from the raw URL and saves it locally.
    In real-world ML pipelines, raw data is downloaded, cached, and versioned.
    """
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
        logger.info(f"Created local data directory: '{LOCAL_DIR}'")
        
    if os.path.exists(LOCAL_FILE):
        logger.info(f"Dataset already cached at '{LOCAL_FILE}'. Skipping network download.")
        return LOCAL_FILE
        
    logger.info(f"Downloading Walmart dataset from URL: {DATA_URL}")
    try:
        # Use python's urllib to fetch the file
        urllib.request.urlretrieve(DATA_URL, LOCAL_FILE)
        logger.info(f"Successfully downloaded and saved dataset to '{LOCAL_FILE}'")
        return LOCAL_FILE
    except Exception as e:
        logger.error(f"Failed to download dataset. Error: {str(e)}")
        raise e

def load_walmart_dataframe() -> pd.DataFrame:
    """
    Loads the cached Walmart dataset into a pandas DataFrame.
    """
    file_path = download_walmart_data()
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded DataFrame with shape: {df.shape}")
        
        # Verify required columns exist
        expected_cols = {'Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'}
        if not expected_cols.issubset(df.columns):
            missing = expected_cols - set(df.columns)
            logger.warning(f"Expected columns are missing from the dataset: {missing}")
            
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV dataset. Error: {str(e)}")
        raise e
