import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

def download_data(ticker: str, start_date: str, end_date: str, 
                 save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Download financial data for a given ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., '^GSPC' for S&P 500)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    save_path : str, optional
        Path to save the downloaded data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the stock data
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Handle MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        # If we have a single ticker, flatten the columns
        if len(data.columns.levels[1]) == 1:
            data.columns = data.columns.droplevel(1)
    
    # Save data if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        print(f"Data saved to {save_path}")
    
    return data

def calculate_returns(data: pd.DataFrame, return_type: str = 'log') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data with 'Adj Close' or 'Close' column
    return_type : str
        Type of returns to calculate ('log' or 'simple')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional returns column
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        # If we have a single ticker, flatten the columns
        if len(df.columns.levels[1]) == 1:
            df.columns = df.columns.droplevel(1)
    
    # Determine which price column to use (Adj Close or Close)
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        raise ValueError(f"DataFrame must contain either 'Adj Close' or 'Close' column. Available columns: {df.columns.tolist()}")
    
    # Ensure the price column has valid numeric data
    try:
        # Convert to numeric if needed
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Check for all NaN values
        if df[price_col].isna().all():
            raise ValueError(f"The {price_col} column contains only NaN values")
    except Exception as e:
        print(f"Error processing price column: {str(e)}")
        print(f"Column type: {type(df[price_col])}")
        print(f"First few values: {df[price_col].head()}")
        raise
    
    # Calculate returns based on type
    try:
        if return_type.lower() == 'log':
            df['return'] = np.log(df[price_col] / df[price_col].shift(1)) * 100
        elif return_type.lower() == 'simple':
            df['return'] = (df[price_col] / df[price_col].shift(1) - 1) * 100
        else:
            raise ValueError("return_type must be either 'log' or 'simple'")
    except Exception as e:
        print(f"Error calculating returns: {str(e)}")
        print(f"First few rows of data:\n{df.head()}")
        raise
    
    # Check if return column was created
    if 'return' not in df.columns:
        raise ValueError("Failed to create 'return' column")
    
    # Drop NaN values resulting from the return calculation
    df = df.dropna(subset=['return'])
    
    # Verify we have data after dropping NaNs
    if len(df) == 0:
        raise ValueError("No valid return data after dropping NaN values")
    
    return df

def check_stationarity(series: pd.Series) -> Dict[str, Any]:
    """
    Check for stationarity of a time series using the Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to check for stationarity
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing ADF test results
    """
    result = adfuller(series.dropna())
    
    output = {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }
    
    return output

def get_crash_period_data(data: pd.DataFrame, period_dict: Dict[str, tuple]) -> Dict[str, pd.DataFrame]:
    """
    Extract data for pre-crash, crash, and post-crash periods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the full dataset
    period_dict : Dict[str, tuple]
        Dictionary with 'pre_crash', 'crash', and 'post_crash' keys, each containing a tuple of (start_date, end_date)
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with 'pre_crash', 'crash', and 'post_crash' keys, each containing the corresponding data
    """
    result = {}
    
    for period_name, period_range in period_dict.items():
        start_date, end_date = period_range
        mask = (data.index >= start_date) & (data.index <= end_date)
        result[period_name] = data[mask].copy()
        
        # Rename 'return' column to 'returns' for consistency
        if 'return' in result[period_name].columns:
            result[period_name]['returns'] = result[period_name]['return']
    
    return result

def plot_returns_volatility(returns: pd.Series, volatility: pd.Series, 
                           title: str = 'Returns and Volatility', 
                           figsize: Tuple[int, int] = (12, 6)):
    """
    Plot returns and volatility together.
    
    Parameters:
    -----------
    returns : pd.Series
        Series containing return data
    volatility : pd.Series
        Series containing volatility data
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot returns
    plt.plot(returns.index, returns, color='blue', alpha=0.5, label='Returns')
    
    # Plot volatility bands
    plt.plot(volatility.index, volatility, color='red', label='Volatility')
    plt.plot(volatility.index, -volatility, color='red')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def prepare_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Complete data preparation pipeline based on configuration.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Dictionary containing configuration parameters
        
    Returns:
    --------
    pd.DataFrame
        Prepared DataFrame with returns
    """
    # Extract config parameters
    ticker = config['ticker']
    start_date = config['start_date']
    end_date = config['end_date']
    return_type = config['return_type']
    
    # Download data
    save_path = f"data/raw_data/{ticker.replace('^', '')}_data.csv"
    data = download_data(ticker, start_date, end_date, save_path)
    
    # Print data info for debugging
    print(f"Downloaded data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"First few rows of data:\n{data.head()}")
    
    # Check for empty data
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker} from {start_date} to {end_date}")
    
    # Calculate returns
    try:
        data_with_returns = calculate_returns(data, return_type)
        print(f"Data with returns shape: {data_with_returns.shape}")
        print(f"Return column stats: min={data_with_returns['return'].min():.2f}, max={data_with_returns['return'].max():.2f}, mean={data_with_returns['return'].mean():.2f}")
        
        # Check for stationarity
        try:
            stationarity_result = check_stationarity(data_with_returns['return'])
            print("Stationarity check:")
            print(f"ADF Statistic: {stationarity_result['adf_statistic']:.4f}")
            print(f"p-value: {stationarity_result['p_value']:.4f}")
            print(f"Is stationary: {stationarity_result['is_stationary']}")
        except Exception as e:
            print(f"Warning: Could not perform stationarity check: {str(e)}")
    except Exception as e:
        print(f"Error calculating returns: {str(e)}")
        # Try to load from saved file as fallback
        if os.path.exists(save_path):
            print(f"Attempting to load data from saved file: {save_path}")
            saved_data = pd.read_csv(save_path, index_col=0, parse_dates=True)
            data_with_returns = calculate_returns(saved_data, return_type)
        else:
            raise
    
    return data_with_returns