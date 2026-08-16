import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import setup_logging

logger = setup_logging()

def build_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the transaction-level Walmart data to a store-level dataset
    specifically for clustering stores based on their sales and environment profiles.
    
    This is a core Data Engineering step in ML pipelines.
    """
    logger.info("Aggregating transactional data to store-level features for clustering.")
    
    # 1. Calculate Holiday sales vs normal sales ratio.
    # This measures how sensitive a store's customers are to holiday promotions.
    holiday_sales = df[df['Holiday_Flag'] == 1].groupby('Store')['Weekly_Sales'].mean()
    normal_sales = df[df['Holiday_Flag'] == 0].groupby('Store')['Weekly_Sales'].mean()
    holiday_ratio = holiday_sales / normal_sales
    
    # 2. Aggregate standard store performance and environmental metrics
    store_agg = df.groupby('Store').agg(
        Mean_Weekly_Sales=('Weekly_Sales', 'mean'),
        Max_Weekly_Sales=('Weekly_Sales', 'max'),
        Std_Weekly_Sales=('Weekly_Sales', 'std'),  # Sales volatility
        Mean_Temperature=('Temperature', 'mean'),
        Mean_Fuel_Price=('Fuel_Price', 'mean'),
        Mean_CPI=('CPI', 'mean'),
        Mean_Unemployment=('Unemployment', 'mean')
    )
    
    # Combine the holiday ratio feature
    store_agg['Holiday_Sales_Ratio'] = holiday_ratio
    # Handle NaN ratios (e.g., if a store had no holiday transactions, set ratio to 1.0)
    store_agg['Holiday_Sales_Ratio'] = store_agg['Holiday_Sales_Ratio'].fillna(1.0)
    
    logger.info(f"Store features aggregated. Total stores: {store_agg.shape[0]}, features: {store_agg.shape[1]}")
    return store_agg

def prepare_forecasting_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering for the demand forecasting (regression) task.
    Extracts time attributes and builds lagged demand variables.
    """
    logger.info("Performing feature engineering for demand forecasting.")
    
    data = df.copy()
    
    # 1. Convert Date column from string to DateTime object
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    
    # 2. Sort data chronologically to maintain time ordering
    data = data.sort_values(by=['Store', 'Date']).reset_index(drop=True)
    
    # 3. Time Feature Extraction
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Week'] = data['Date'].dt.isocalendar().week.astype(int)
    
    # 4. Lagged Feature (Last week's sales): Crucial for time-series forecasting.
    # Group by Store to make sure we don't mix sales between different stores!
    data['Weekly_Sales_Lag1'] = data.groupby('Store')['Weekly_Sales'].shift(1)
    
    # 5. Drop rows with missing values (the first week of each store has no previous week)
    initial_len = len(data)
    data = data.dropna().reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(data)} rows with NaN lagged values. Total records: {len(data)}")
    
    return data

def train_test_split_chronological(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Splits the dataset into train and test sets chronologically.
    This is essential in time-series to avoid 'lookahead bias' or 'data leakage'.
    """
    logger.info(f"Splitting data chronologically with train_ratio={train_ratio}")
    
    # Sort and find the split boundary date
    unique_dates = sorted(data['Date'].unique())
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]
    
    # Split the DataFrame
    train_df = data[data['Date'] < split_date]
    test_df = data[data['Date'] >= split_date]
    
    logger.info(f"Split done. Train records: {train_df.shape[0]}, Test records: {test_df.shape[0]}")
    logger.info(f"Train date range: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
    logger.info(f"Test date range: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
    
    return train_df, test_df
