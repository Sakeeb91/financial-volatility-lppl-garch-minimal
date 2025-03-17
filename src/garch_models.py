import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GARCHModel:
    """
    Wrapper class for GARCH models using the arch package.
    """
    def __init__(self, p: int = 1, q: int = 1, o: int = 0, mean: str = 'Zero', vol: str = 'GARCH'):
        """
        Initialize GARCH model.
        
        Args:
            p (int): GARCH lag order
            q (int): ARCH lag order
            o (int): Asymmetry order (for GJR-GARCH)
            mean (str): Mean model specification
            vol (str): Volatility model specification ('GARCH', 'EGARCH', 'GJRGARCH', etc.)
        """
        self.p = p
        self.q = q
        self.o = o
        self.mean = mean
        self.vol = vol
        self.model = None
        self.results = None
        self.forecasts = None
        
    def fit(self, returns: np.ndarray) -> None:
        """
        Fit the GARCH model to the returns data.
        
        Args:
            returns (np.ndarray): Array of returns
        """
        self.model = arch_model(returns, p=self.p, q=self.q, o=self.o,
                              mean=self.mean, vol=self.vol)
        self.results = self.model.fit(disp='off')

    def get_conditional_volatility(self) -> pd.Series:
        """
        Get the conditional volatility from the fitted model.
        
        Returns:
        --------
        pd.Series
            Conditional volatility series
        """
        if self.results is None:
            raise ValueError("Model must be fit before getting conditional volatility")
            
        volatility = self.results.conditional_volatility
        return volatility
    
    def forecast(self, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts for mean and variance.
        
        Args:
            horizon (int): Forecast horizon
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and variance forecasts
        """
        if self.results is None:
            raise ValueError("Model must be fit before forecasting")
        
        forecast = self.results.forecast(horizon=horizon)
        
        # Extract forecasts based on horizon
        if horizon == 1:
            # For 1-step ahead forecast, get the first forecast
            mean_forecast = forecast.mean.iloc[0].values
            variance_forecast = forecast.variance.iloc[0].values
        else:
            # For multi-step forecasts, get all forecasts
            mean_forecast = forecast.mean.values
            variance_forecast = forecast.variance.values
            
        return mean_forecast, variance_forecast

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and diagnostics.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        if self.results is None:
            raise ValueError("Model must be fit before getting information")
        
        return {
            'aic': self.results.aic,
            'bic': self.results.bic,
            'params': self.results.params,
            'pvalues': self.results.pvalues,
            'loglikelihood': self.results.loglikelihood
        }

    def evaluate(self, actual: pd.Series, forecast_type: str = 'variance') -> Dict[str, float]:
        """
        Evaluate forecast performance against actual values.
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values (typically squared returns as volatility proxy)
        forecast_type : str
            Type of forecast to evaluate ('variance' or 'volatility')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        if self.forecasts is None:
            raise ValueError("Model must be forecast before evaluation")
            
        # Extract forecasts
        if forecast_type == 'variance':
            forecasts = self.forecasts.variance.values[-1, :]
        elif forecast_type == 'volatility':
            forecasts = np.sqrt(self.forecasts.variance.values[-1, :])
        else:
            raise ValueError("forecast_type must be 'variance' or 'volatility'")
            
        # Calculate metrics
        mse = mean_squared_error(actual, forecasts)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, forecasts)
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero by adding small epsilon
        mape = np.mean(np.abs((actual - forecasts) / (actual + 1e-10))) * 100
        
        # Directional accuracy
        actual_dir = np.sign(np.diff(np.concatenate(([actual[0]], actual))))
        forecast_dir = np.sign(np.diff(np.concatenate(([forecasts[0]], forecasts))))
        dir_accuracy = np.mean(actual_dir == forecast_dir) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'dir_accuracy': dir_accuracy
        }
    
    def plot_diagnostics(self, returns: pd.Series) -> None:
        """
        Plot model diagnostics.
        
        Parameters:
        -----------
        returns : pd.Series
            Original return series used for fitting
        """
        if self.results is None:
            raise ValueError("Model must be fit before plotting diagnostics")
            
        # Get conditional volatility
        volatility = self.get_conditional_volatility()
        
        # Plot returns with volatility bands
        plt.figure(figsize=(12, 6))
        plt.plot(returns.index, returns, color='blue', alpha=0.5, label='Returns')
        plt.plot(volatility.index, volatility, color='red', label='Conditional Volatility')
        plt.plot(volatility.index, -volatility, color='red')
        plt.title(f"{self.vol} Model: Returns with Volatility Bands")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Standardized residuals
        std_resid = returns / volatility
        
        # Plot standardized residuals
        plt.figure(figsize=(12, 6))
        plt.plot(std_resid.index, std_resid)
        plt.title('Standardized Residuals')
        plt.tight_layout()
        plt.show()
        
        # QQ Plot of standardized residuals
        import statsmodels.api as sm
        from scipy import stats
        
        plt.figure(figsize=(10, 6))
        sm.qqplot(std_resid, stats.norm, line='45', fit=True)
        plt.title('QQ Plot of Standardized Residuals')
        plt.tight_layout()
        plt.show()
        
        # ACF of squared standardized residuals
        from statsmodels.graphics.tsaplots import plot_acf
        
        plt.figure(figsize=(12, 6))
        plot_acf(std_resid**2, lags=20)
        plt.title('ACF of Squared Standardized Residuals')
        plt.tight_layout()
        plt.show()

def compare_garch_models(returns: pd.Series, 
                        model_configs: Dict[str, Dict[str, Any]], 
                        plot: bool = True) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Compare multiple GARCH model specifications.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series for fitting models
    model_configs : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to configuration dictionaries
    plot : bool
        Whether to plot comparison results
        
    Returns:
    --------
    Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]
        Dictionary containing model results and DataFrame of volatilities
    """
    results = {}
    volatilities = pd.DataFrame(index=returns.index)
    
    # Fit all models
    for name, config in model_configs.items():
        print(f"Fitting {name} model with config: {config}")
        
        # Extract parameters with defaults
        p = config.get('p', 1)
        q = config.get('q', 1)
        o = config.get('o', 0)
        mean = config.get('mean', 'Zero')
        vol = config.get('vol', 'GARCH')
        
        print(f"Using parameters: p={p}, q={q}, o={o}, mean={mean}, vol={vol}")
        
        model = GARCHModel(
            p=p,
            q=q,
            o=o,
            mean=mean,
            vol=vol
        )
        model.fit(returns)
        
        # Store results
        results[name] = {
            'model': model,
            'fit_result': model.results,
            'aic': model.results.aic,
            'bic': model.results.bic,
            'log_likelihood': model.results.loglikelihood
        }
        
        # Store conditional volatility
        volatilities[name] = model.results.conditional_volatility
    
    # Plot comparison if requested
    if plot:
        plt.figure(figsize=(12, 8))
        for column in volatilities.columns:
            plt.plot(volatilities.index, volatilities[column], label=column)
        plt.title('Comparison of GARCH Model Volatilities')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return results, volatilities

def rolling_window_forecast(returns: pd.Series, 
                           model_config: Dict[str, Any],
                           window_size: int = 252, 
                           step_size: int = 21,
                           horizon: int = 1,
                           volatility_proxy: str = 'squared_returns') -> pd.DataFrame:
    """
    Perform rolling window forecasting and evaluation.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    model_config : Dict[str, Any]
        GARCH model configuration
    window_size : int
        Size of rolling window (days)
    step_size : int
        Step size for rolling (days)
    horizon : int
        Forecast horizon (days)
    volatility_proxy : str
        Proxy for actual volatility ('squared_returns' or 'abs_returns')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing evaluation metrics for each window
    """
    results = []
    
    # Convert returns to numpy for faster processing
    returns_np = returns.values
    dates = returns.index
    
    # Number of windows
    n_windows = (len(returns) - window_size - horizon) // step_size + 1
    
    for i in range(n_windows):
        # Define window indices
        start_idx = i * step_size
        train_end_idx = start_idx + window_size
        test_end_idx = train_end_idx + horizon
        
        # Skip if we don't have enough data
        if test_end_idx > len(returns):
            break
            
        # Extract training and test data
        train_returns = returns.iloc[start_idx:train_end_idx]
        test_returns = returns.iloc[train_end_idx:test_end_idx]
        
        # Fit model on training data
        model = GARCHModel(
            p=model_config.get('p', 1),
            q=model_config.get('q', 1),
            o=model_config.get('o', 0),
            mean=model_config.get('mean', 'Zero'),
            vol=model_config.get('vol', 'GARCH')
        )
        model.fit(train_returns)
        
        # Forecast for test period
        forecasts = model.forecast(horizon=horizon)
        forecast_variance = forecasts.variance.values[-1, :]
        
        # Define actual volatility proxy
        if volatility_proxy == 'squared_returns':
            actual_vol = test_returns.values**2
        elif volatility_proxy == 'abs_returns':
            actual_vol = np.abs(test_returns.values)
        else:
            raise ValueError("volatility_proxy must be 'squared_returns' or 'abs_returns'")
        
        # Calculate metrics
        mse = mean_squared_error(actual_vol, forecast_variance)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_vol, forecast_variance)
        
        # Store results
        results.append({
            'start_date': dates[start_idx],
            'train_end_date': dates[train_end_idx - 1],
            'test_end_date': dates[test_end_idx - 1],
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        })
    
    return pd.DataFrame(results)