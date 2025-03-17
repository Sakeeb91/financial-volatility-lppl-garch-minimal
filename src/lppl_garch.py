import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional, Union
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import custom models
from src.lppl_models import LPPLModel

class LPPLGARCHModel:
    """
    Custom GARCH model incorporating Log-Periodic Power Law (LPPL) components.
    
    This hybrid model enhances traditional GARCH by including a log-periodic
    term in the volatility equation, potentially improving forecasts during
    market crashes and bubble regimes.
    """
    
    def __init__(self, garch_params: Dict[str, Any] = None, lppl_bounds: Dict[str, Tuple[float, float]] = None):
        """
        Initialize the LPPL-GARCH model.
        
        Parameters:
        -----------
        garch_params : Dict[str, Any], optional
            GARCH model parameters (p, q, o, mean, vol)
        lppl_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for LPPL parameters
        """
        # Default GARCH parameters
        self.garch_params = garch_params if garch_params is not None else {
            'p': 1,
            'q': 1,
            'o': 0,
            'mean': 'Zero',
            'vol': 'Garch'
        }
        
        # Initialize LPPL model
        self.lppl_model = LPPLModel(bounds=lppl_bounds)
        
        # Storage for model components
        self.garch_vol = None
        self.lppl_component = None
        self.combined_vol = None
        self.lppl_weight = 0.5  # Default weight for LPPL component
        self.fit_result = None
        
    def _calculate_lppl_volatility(self, returns: pd.Series, window_size: int = 252) -> pd.Series:
        """
        Calculate volatility component from LPPL model.
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        window_size : int
            Window size for rolling LPPL estimation
            
        Returns:
        --------
        pd.Series
            LPPL volatility component
        """
        # Convert returns to prices (start with 100)
        prices = 100 * (1 + returns / 100).cumprod()
        
        # Initialize LPPL volatility series
        lppl_vol = pd.Series(np.nan, index=returns.index)
        
        # Calculate for each window
        for i in range(window_size, len(returns), window_size // 4):
            # Define window
            start_idx = max(0, i - window_size)
            end_idx = i
            
            # Extract window
            window_prices = prices.iloc[start_idx:end_idx]
            
            try:
                # Fit LPPL model
                self.lppl_model.fit(window_prices)
                
                # Extract parameters
                params = self.lppl_model.params
                
                # Only use if bubble regime parameters
                is_bubble = (0 < params.m < 1 and params.B < 0)
                
                if is_bubble:
                    # Calculate time to critical point
                    t = np.arange(len(window_prices))
                    tc = params.tc
                    dt = tc - t
                    
                    # Calculate volatility component
                    # Higher as we approach tc
                    lppl_component = np.zeros(len(window_prices))
                    valid_idx = np.where((dt > 0) & (dt <= tc))[0]
                    
                    if len(valid_idx) > 0:
                        # Amplitude increases as we approach tc
                        lppl_component[valid_idx] = np.abs(params.B) * (dt[valid_idx]**params.m) * (
                            1 + params.C1 * np.cos(params.omega * np.log(dt[valid_idx])) + 
                            params.C2 * np.sin(params.omega * np.log(dt[valid_idx]))
                        )
                        
                        # Scale to volatility range (0-20%)
                        max_vol = 20
                        lppl_component = np.minimum(lppl_component, max_vol)
                        
                        # Assign to output series
                        lppl_vol.iloc[start_idx:end_idx] = lppl_component
            except:
                # Skip failed fits
                continue
        
        # Forward fill NaN values
        lppl_vol = lppl_vol.fillna(method='ffill').fillna(0)
        
        return lppl_vol
    
    def fit(self, returns: pd.Series, lppl_window: int = 252, lppl_weight: float = 0.5) -> Any:
        """
        Fit the LPPL-GARCH model to return data.
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        lppl_window : int
            Window size for LPPL component estimation
        lppl_weight : float
            Weight of LPPL component in combined volatility (0-1)
            
        Returns:
        --------
        Any
            Fit results
        """
        # Store LPPL weight
        self.lppl_weight = lppl_weight
        
        # Calculate LPPL volatility component
        self.lppl_component = self._calculate_lppl_volatility(returns, lppl_window)
        
        # Fit standard GARCH model
        garch_model = arch_model(
            returns,
            p=self.garch_params.get('p', 1),
            o=self.garch_params.get('o', 0),
            q=self.garch_params.get('q', 1),
            mean=self.garch_params.get('mean', 'Zero'),
            vol=self.garch_params.get('vol', 'Garch')
        )
        
        self.fit_result = garch_model.fit(disp='off')
        
        # Extract GARCH volatility
        self.garch_vol = self.fit_result.conditional_volatility
        
        # Combine volatilities
        self.combined_vol = self.garch_vol * (1 - lppl_weight) + self.lppl_component * lppl_weight
        
        return self.fit_result
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Get the combined conditional volatility.
        
        Returns:
        --------
        pd.Series
            Combined volatility series
        """
        if self.combined_vol is None:
            raise ValueError("Model must be fit before getting volatility")
        
        return self.combined_vol
    
    def forecast(self, horizon: int = 1, start: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate forecasts for volatility.
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon
        start : Optional[int]
            Start index for forecasting
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing forecast results
        """
        if self.fit_result is None:
            raise ValueError("Model must be fit before forecasting")
        
        # Generate GARCH forecasts
        garch_forecasts = self.fit_result.forecast(horizon=horizon, start=start)
        
        # Extract last LPPL component value
        last_lppl = self.lppl_component.iloc[-1]
        
        # For LPPL component, we simply extend the last value
        # In a more sophisticated model, you would use the LPPL parameters to project forward
        lppl_forecast = np.ones(horizon) * last_lppl
        
        # Combine forecasts
        combined_forecast = (
            garch_forecasts.variance.values[-1, :] * (1 - self.lppl_weight) + 
            lppl_forecast * self.lppl_weight
        )
        
        return {
            'garch': garch_forecasts.variance.values[-1, :],
            'lppl': lppl_forecast,
            'combined': combined_forecast,
            'variance': combined_forecast  # Add this key for compatibility with comparison script
        }
    
    def evaluate(self, actual: pd.Series, forecasts: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecast performance against actual values.
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values (typically squared returns as volatility proxy)
        forecasts : Dict[str, np.ndarray]
            Dictionary of forecast arrays
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of evaluation metrics for each model
        """
        results = {}
        
        for model_name, forecast_values in forecasts.items():
            # Calculate metrics
            mse = mean_squared_error(actual, forecast_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, forecast_values)
            
            # Mean Absolute Percentage Error (MAPE)
            # Avoid division by zero
            mape = np.mean(np.abs((actual - forecast_values) / (actual + 1e-10))) * 100
            
            results[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
        
        return results
    
    def plot_components(self, returns: pd.Series, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot volatility components and combined volatility.
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        figsize : Tuple[int, int]
            Figure size
        """
        if self.combined_vol is None:
            raise ValueError("Model must be fit before plotting")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot returns
        plt.plot(returns.index, returns, color='blue', alpha=0.3, label='Returns')
        
        # Plot volatility components
        plt.plot(self.garch_vol.index, self.garch_vol, color='green', label='GARCH Volatility')
        plt.plot(self.lppl_component.index, self.lppl_component, color='red', label='LPPL Component')
        plt.plot(self.combined_vol.index, self.combined_vol, color='purple', 
                linewidth=2, label='Combined Volatility')
        
        plt.title('LPPL-GARCH Model Components')
        plt.xlabel('Date')
        plt.ylabel('Volatility / Returns')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot LPPL component contribution to total volatility
        plt.figure(figsize=figsize)
        
        contribution = (self.lppl_component / self.combined_vol) * 100
        plt.plot(contribution.index, contribution)
        plt.axhline(y=50, color='red', linestyle='--')
        
        plt.title('LPPL Component Contribution to Total Volatility (%)')
        plt.xlabel('Date')
        plt.ylabel('Contribution (%)')
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(self, returns: pd.Series, forecasts: Dict[str, np.ndarray], 
                    horizon: int, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot forecasted volatility.
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        forecasts : Dict[str, np.ndarray]
            Dictionary of forecast arrays
        horizon : int
            Forecast horizon
        figsize : Tuple[int, int]
            Figure size
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create forecast dates
        last_date = returns.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
        
        # Plot historical volatility
        plt.plot(self.combined_vol.index[-30:], self.combined_vol.iloc[-30:], 
                color='blue', label='Historical Volatility')
        
        # Plot forecasts
        colors = {'garch': 'green', 'lppl': 'red', 'combined': 'purple'}
        
        for model_name, forecast_values in forecasts.items():
            plt.plot(forecast_dates, forecast_values, color=colors.get(model_name, 'gray'), 
                    label=f'{model_name.upper()} Forecast')
        
        plt.axvline(x=last_date, color='black', linestyle='--', label='Forecast Start')
        
        plt.title('Volatility Forecasts')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.tight_layout()
        plt.show()

def compare_models(returns: pd.Series, 
                 garch_params: Dict[str, Any] = None,
                 lppl_bounds: Dict[str, Tuple[float, float]] = None,
                 lppl_weights: List[float] = [0.3, 0.5, 0.7],
                 test_size: int = 21,
                 plot: bool = True) -> Dict[str, Any]:
    """
    Compare standard GARCH with LPPL-GARCH models using different weights.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    garch_params : Dict[str, Any], optional
        GARCH model parameters
    lppl_bounds : Dict[str, Tuple[float, float]], optional
        Bounds for LPPL parameters
    lppl_weights : List[float]
        List of weights to test for LPPL component
    test_size : int
        Size of test set
    plot : bool
        Whether to plot comparison results
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary of comparison results
    """
    # Split data
    train_returns = returns.iloc[:-test_size]
    test_returns = returns.iloc[-test_size:]
    
    results = {}
    forecasts = {}
    
    # Fit standard GARCH model
    garch_model = arch_model(
        train_returns,
        p=garch_params.get('p', 1) if garch_params else 1,
        o=garch_params.get('o', 0) if garch_params else 0,
        q=garch_params.get('q', 1) if garch_params else 1,
        mean=garch_params.get('mean', 'Zero') if garch_params else 'Zero',
        vol=garch_params.get('vol', 'Garch') if garch_params else 'Garch'
    )
    
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=test_size)
    
    results['garch'] = {
        'model': garch_model,
        'fit': garch_fit,
        'forecast': garch_forecast
    }
    
    forecasts['garch'] = garch_forecast.variance.values[-1, :]
    
    # Fit LPPL-GARCH models with different weights
    for weight in lppl_weights:
        model_name = f'lppl_garch_{weight}'
        
        lppl_garch = LPPLGARCHModel(garch_params=garch_params, lppl_bounds=lppl_bounds)
        lppl_garch.fit(train_returns, lppl_weight=weight)
        
        lppl_forecast = lppl_garch.forecast(horizon=test_size)
        
        results[model_name] = {
            'model': lppl_garch,
            'forecast': lppl_forecast
        }
        
        forecasts[model_name] = lppl_forecast['combined']
    
    # Evaluate forecasts
    actual_vol = test_returns**2  # Squared returns as volatility proxy
    
    eval_results = {}
    
    for model_name, forecast_values in forecasts.items():
        mse = mean_squared_error(actual_vol, forecast_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_vol, forecast_values)
        
        eval_results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    # Plot if requested
    if plot:
        # Plot forecasts
        plt.figure(figsize=(12, 6))
        
        # Create forecast dates
        last_date = train_returns.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(test_size)]
        
        colors = {
            'garch': 'green',
            f'lppl_garch_{lppl_weights[0]}': 'red',
            f'lppl_garch_{lppl_weights[1]}': 'purple',
            f'lppl_garch_{lppl_weights[2]}': 'orange'
        }
        
        for model_name, forecast_values in forecasts.items():
            plt.plot(forecast_dates, forecast_values, 
                    color=colors.get(model_name, 'gray'),
                    label=f'{model_name.upper()} Forecast')
        
        # Plot

def rolling_window_comparison(returns: pd.Series,
                           window_size: int = 252,
                           step_size: int = 21,
                           garch_params: Dict[str, Any] = None,
                           lppl_bounds: Dict[str, Tuple[float, float]] = None,
                           lppl_weight: float = 0.5,
                           forecast_horizon: int = 21) -> Dict[str, Any]:
    """
    Perform rolling window comparison of GARCH and LPPL-GARCH models.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    window_size : int
        Size of rolling window
    step_size : int
        Number of steps to move window forward
    garch_params : Dict[str, Any], optional
        GARCH model parameters
    lppl_bounds : Dict[str, Tuple[float, float]], optional
        Bounds for LPPL parameters
    lppl_weight : float
        Weight for LPPL component
    forecast_horizon : int
        Forecast horizon for each window
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing rolling window results
    """
    results = {
        'garch': {'mse': [], 'rmse': [], 'mae': []},
        'lppl_garch': {'mse': [], 'rmse': [], 'mae': []}
    }
    
    # Initialize dates for storing results
    eval_dates = []
    
    # Roll through the data
    for start_idx in range(0, len(returns) - window_size - forecast_horizon, step_size):
        # Define window
        end_idx = start_idx + window_size
        train_returns = returns.iloc[start_idx:end_idx]
        test_returns = returns.iloc[end_idx:end_idx + forecast_horizon]
        
        # Store evaluation date
        eval_dates.append(returns.index[end_idx])
        
        # Fit standard GARCH
        garch_model = arch_model(
            train_returns,
            p=garch_params.get('p', 1) if garch_params else 1,
            o=garch_params.get('o', 0) if garch_params else 0,
            q=garch_params.get('q', 1) if garch_params else 1,
            mean=garch_params.get('mean', 'Zero') if garch_params else 'Zero',
            vol=garch_params.get('vol', 'Garch') if garch_params else 'Garch'
        )
        
        garch_fit = garch_model.fit(disp='off')
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon)
        
        # Fit LPPL-GARCH
        lppl_garch = LPPLGARCHModel(garch_params=garch_params, lppl_bounds=lppl_bounds)
        lppl_garch.fit(train_returns, lppl_weight=lppl_weight)
        lppl_forecast = lppl_garch.forecast(horizon=forecast_horizon)
        
        # Calculate metrics
        actual_vol = test_returns**2
        
        # GARCH metrics
        garch_mse = mean_squared_error(actual_vol, garch_forecast.variance.values[-1, :])
        results['garch']['mse'].append(garch_mse)
        results['garch']['rmse'].append(np.sqrt(garch_mse))
        results['garch']['mae'].append(mean_absolute_error(actual_vol, garch_forecast.variance.values[-1, :]))
        
        # LPPL-GARCH metrics
        lppl_mse = mean_squared_error(actual_vol, lppl_forecast['combined'])
        results['lppl_garch']['mse'].append(lppl_mse)
        results['lppl_garch']['rmse'].append(np.sqrt(lppl_mse))
        results['lppl_garch']['mae'].append(mean_absolute_error(actual_vol, lppl_forecast['combined']))
    
    # Convert results to DataFrames
    for model in results:
        for metric in results[model]:
            results[model][metric] = pd.Series(results[model][metric], index=eval_dates)
    
    return results