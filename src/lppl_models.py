import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple, List, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class LPPLParams:
    """
    Data class for LPPL model parameters.
    """
    tc: float        # Critical time (time of crash)
    m: float         # Power law exponent
    omega: float     # Log-periodic frequency
    A: float         # Constant
    B: float         # Power law amplitude
    C1: float        # Cosine amplitude
    C2: float        # Sine amplitude
    
    def as_array(self) -> np.ndarray:
        """Convert parameters to numpy array."""
        return np.array([self.tc, self.m, self.omega, self.A, self.B, self.C1, self.C2])
    
    @classmethod
    def from_array(cls, params: np.ndarray) -> 'LPPLParams':
        """Create LPPLParams from numpy array."""
        return cls(
            tc=params[0],
            m=params[1],
            omega=params[2],
            A=params[3],
            B=params[4],
            C1=params[5],
            C2=params[6]
        )

class LPPLModel:
    """
    Log-Periodic Power Law model implementation.
    """
    def __init__(self, bounds: Dict[str, Tuple[float, float]] = None):
        """
        Initialize LPPL model with parameter bounds.
        
        Parameters:
        -----------
        bounds : Dict[str, Tuple[float, float]], optional
            Dictionary of parameter bounds
        """
        # Default parameter bounds
        self.default_bounds = {
            'tc': (0, 50),       # Will be adjusted based on data
            'm': (0.1, 0.9),     # 0 < m < 1 for crashes
            'omega': (6.0, 13.0), # Typical range for log-periodic oscillations
            'A': (-1000, 1000),
            'B': (-1000, 1000),
            'C1': (-1000, 1000),
            'C2': (-1000, 1000)
        }
        
        self.bounds = bounds if bounds is not None else self.default_bounds
        self.params = None
        self.fitted_values = None
        self.residuals = None
        
    def lppl_function(self, t: np.ndarray, params: LPPLParams) -> np.ndarray:
        """
        Calculate LPPL function values for given time points and parameters.
        
        Parameters:
        -----------
        t : np.ndarray
            Time array
        params : LPPLParams
            LPPL parameters
            
        Returns:
        --------
        np.ndarray
            LPPL function values
        """
        # Time to critical time
        dt = params.tc - t
        
        # Filter valid dt values (must be positive)
        valid_mask = dt > 0
        if not any(valid_mask):
            return np.ones_like(t) * np.inf
        
        result = np.ones_like(t) * np.inf
        valid_dt = dt[valid_mask]
        
        # Calculate LPPL components
        power_law = params.A + params.B * valid_dt**params.m
        log_periodic = (params.C1 * np.cos(params.omega * np.log(valid_dt)) + 
                        params.C2 * np.sin(params.omega * np.log(valid_dt))) * valid_dt**params.m
        
        # Combine components
        result[valid_mask] = power_law + log_periodic
        
        return result
    
    def cost_function(self, params_array: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        """
        Cost function for LPPL parameter optimization.
        
        Parameters:
        -----------
        params_array : np.ndarray
            Array of LPPL parameters
        t : np.ndarray
            Time array
        y : np.ndarray
            Target values (typically log prices)
            
        Returns:
        --------
        float
            Sum of squared errors
        """
        params = LPPLParams.from_array(params_array)
        y_pred = self.lppl_function(t, params)
        
        # Mask for valid predictions
        valid_mask = np.isfinite(y_pred)
        if not any(valid_mask):
            return np.inf
        
        # Calculate sum of squared errors
        sse = np.sum((y[valid_mask] - y_pred[valid_mask])**2)
        
        return sse
    
    def fit(self, prices: pd.Series, t: Optional[np.ndarray] = None, 
            method: str = 'differential_evolution', 
            refine: bool = True,
            refine_method: str = 'Nelder-Mead',
            log_transform: bool = True,
            **kwargs) -> LPPLParams:
        """
        Fit LPPL model to price data.
        
        Parameters:
        -----------
        prices : pd.Series
            Series of price data
        t : np.ndarray, optional
            Time array (if None, will create array counting days from start)
        method : str
            Optimization method ('differential_evolution' or 'minimize')
        refine : bool
            Whether to refine the solution with a second optimization
        refine_method : str
            Method for refinement optimization
        log_transform : bool
            Whether to log-transform prices before fitting
        **kwargs
            Additional arguments for the optimizer
            
        Returns:
        --------
        LPPLParams
            Optimized LPPL parameters
        """
        # Create time array if not provided
        if t is None:
            t = np.arange(len(prices))
        
        # Transform prices if requested
        if log_transform:
            y = np.log(prices.values)
        else:
            y = prices.values
        
        # Adjust tc bounds based on time array
        bounds_list = []
        for param_name, (lower, upper) in self.bounds.items():
            if param_name == 'tc':
                # tc should be greater than max(t)
                bounds_list.append((t.max(), t.max() + upper))
            else:
                bounds_list.append((lower, upper))
        
        # Optimize parameters
        if method == 'differential_evolution':
            result = differential_evolution(
                self.cost_function,
                bounds_list,
                args=(t, y),
                **kwargs
            )
        elif method == 'minimize':
            # For minimize, we need an initial guess
            initial_guess = [
                t.max() + 10,  # tc: crash time estimate
                0.5,           # m: middle of allowed range
                8.0,           # omega: typical value
                np.mean(y),    # A: mean of log prices
                -0.01,         # B: small negative value
                0.01,          # C1: small positive value
                0.01           # C2: small positive value
            ]
            
            result = minimize(
                self.cost_function,
                initial_guess,
                args=(t, y),
                bounds=bounds_list,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Store parameters
        self.params = LPPLParams.from_array(result.x)
        
        # Refine solution if requested
        if refine and result.success:
            refined_result = minimize(
                self.cost_function,
                result.x,
                args=(t, y),
                method=refine_method
            )
            
            if refined_result.success:
                self.params = LPPLParams.from_array(refined_result.x)
        
        # Calculate fitted values and residuals
        self.fitted_values = self.lppl_function(t, self.params)
        self.residuals = y - self.fitted_values
        
        return self.params
    
    def plot_fit(self, prices: pd.Series, t: Optional[np.ndarray] = None,
                log_transform: bool = True, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot original data with fitted LPPL model.
        
        Parameters:
        -----------
        prices : pd.Series
            Series of price data
        t : np.ndarray, optional
            Time array (if None, will use index)
        log_transform : bool
            Whether to log-transform prices for visualization
        figsize : Tuple[int, int]
            Figure size
        """
        if self.params is None or self.fitted_values is None:
            raise ValueError("Model must be fit before plotting")
        
        # Create time array if not provided
        if t is None:
            t = np.arange(len(prices))
        
        plt.figure(figsize=figsize)
        
        # Plot original data
        if log_transform:
            plt.plot(prices.index, np.log(prices), label='Log Prices', color='blue')
            plt.ylabel('Log Price')
        else:
            plt.plot(prices.index, prices, label='Prices', color='blue')
            plt.ylabel('Price')
        
        # Plot fitted values
        valid_mask = np.isfinite(self.fitted_values)
        if log_transform:
            plt.plot(prices.index[valid_mask], self.fitted_values[valid_mask],
                   label='LPPL Fit', color='red', linestyle='--')
        else:
            plt.plot(prices.index[valid_mask], np.exp(self.fitted_values[valid_mask]),
                   label='LPPL Fit', color='red', linestyle='--')
        
        # Add critical time marker
        if isinstance(prices.index, pd.DatetimeIndex):
            tc_date = prices.index[0] + pd.Timedelta(days=int(self.params.tc - t[0]))
            plt.axvline(x=tc_date, color='green', linestyle='--', 
                       label=f'Critical Time (tc = {tc_date.strftime("%Y-%m-%d")})')
        else:
            plt.axvline(x=self.params.tc, color='green', linestyle='--', 
                       label=f'Critical Time (tc = {self.params.tc:.2f})')
        
        # Add parameter annotations
        params_text = '\n'.join([
            f"m = {self.params.m:.3f}",
            f"omega = {self.params.omega:.3f}",
            f"A = {self.params.A:.3f}",
            f"B = {self.params.B:.3f}",
            f"C1 = {self.params.C1:.3f}",
            f"C2 = {self.params.C2:.3f}"
        ])
        plt.annotate(params_text, xy=(0.02, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.title('LPPL Model Fit')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def is_bubble_regime(self) -> bool:
        """
        Check if the fitted parameters indicate a bubble regime.
        
        Returns:
        --------
        bool
            True if parameters indicate a bubble regime, False otherwise
        """
        if self.params is None:
            raise ValueError("Model must be fit before checking bubble regime")
        
        # Conditions for bubble regime:
        # 1. 0 < m < 1 (power law exponent)
        # 2. B < 0 (negative power law amplitude)
        # 3. omega > 0 (positive log-periodic frequency)
        
        return (0 < self.params.m < 1 and 
                self.params.B < 0 and 
                self.params.omega > 0)
    
    def extend_forecast(self, prices: pd.Series, days_forward: int = 30, 
                      log_transform: bool = True, plot: bool = True) -> pd.Series:
        """
        Extend LPPL forecast into the future.
        
        Parameters:
        -----------
        prices : pd.Series
            Original price series
        days_forward : int
            Number of days to forecast forward
        log_transform : bool
            Whether to log-transform prices
        plot : bool
            Whether to plot the forecast
            
        Returns:
        --------
        pd.Series
            Series containing the forecasted values
        """
        if self.params is None:
            raise ValueError("Model must be fit before forecasting")
        
        # Create extended time array
        t_orig = np.arange(len(prices))
        t_future = np.arange(len(prices), len(prices) + days_forward)
        t_all = np.concatenate([t_orig, t_future])
        
        # Generate forecasts
        lppl_values = self.lppl_function(t_all, self.params)
        
        # Transform if needed
        if log_transform:
            forecast_values = np.exp(lppl_values)
        else:
            forecast_values = lppl_values
            
        # Create date index for forecasts
        last_date = prices.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_forward)]
        all_dates = pd.DatetimeIndex(list(prices.index) + future_dates)
        
        # Create forecast series
        forecast_series = pd.Series(forecast_values, index=all_dates)
        
        # Plot if requested
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(prices.index, prices, label='Actual', color='blue')
            
            # Plot fitted and forecast values
            valid_mask = np.isfinite(forecast_values)
            plt.plot(all_dates[valid_mask], forecast_values[valid_mask], 
                    label='LPPL Fit + Forecast', color='red', linestyle='--')
            
            # Add vertical line separating history from forecast
            plt.axvline(x=last_date, color='green', linestyle='--', 
                       label='Forecast Start')
            
            # Add critical time marker
            tc_date = prices.index[0] + pd.Timedelta(days=int(self.params.tc))
            plt.axvline(x=tc_date, color='purple', linestyle=':', 
                       label=f'Critical Time (tc)')
            
            plt.title('LPPL Model Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        return forecast_series
    
    def get_confidence_interval(self, prices: pd.Series, t: Optional[np.ndarray] = None,
                             n_bootstrap: int = 100, confidence: float = 0.95,
                             log_transform: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate confidence intervals for LPPL forecasts using bootstrap.
        
        Parameters:
        -----------
        prices : pd.Series
            Price data
        t : np.ndarray, optional
            Time array
        n_bootstrap : int
            Number of bootstrap samples
        confidence : float
            Confidence level (0-1)
        log_transform : bool
            Whether to log-transform prices
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with lower and upper confidence bounds
        """
        if self.params is None:
            raise ValueError("Model must be fit before generating confidence intervals")
        
        # Create time array if not provided
        if t is None:
            t = np.arange(len(prices))
        
        # Transform prices if requested
        if log_transform:
            y = np.log(prices.values)
        else:
            y = prices.values
        
        # Original fitted values
        original_fit = self.fitted_values
        
        # Storage for bootstrap samples
        bootstrap_fits = np.zeros((n_bootstrap, len(t)))
        
        # Generate bootstrap samples
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_bootstrap):
            # Resample residuals
            bootstrap_residuals = np.random.choice(
                self.residuals[np.isfinite(self.residuals)], 
                size=len(y),
                replace=True
            )
            
            # Create bootstrap sample
            bootstrap_y = original_fit.copy()
            bootstrap_y[np.isfinite(bootstrap_y)] += bootstrap_residuals
            
            # Fit model to bootstrap sample
            model = LPPLModel(bounds=self.bounds)
            try:
                model.fit(pd.Series(np.exp(bootstrap_y) if log_transform else bootstrap_y), 
                        t=t, log_transform=log_transform)
                bootstrap_fits[i] = model.fitted_values
            except:
                # If fit fails, use original fit
                bootstrap_fits[i] = original_fit
        
        # Calculate confidence intervals
        alpha = (1 - confidence) / 2
        lower_bound = np.nanpercentile(bootstrap_fits, alpha * 100, axis=0)
        upper_bound = np.nanpercentile(bootstrap_fits, (1 - alpha) * 100, axis=0)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound
        }

def find_crash_signals(prices: pd.Series, window_sizes: List[int] = [252, 500], 
                     min_confidence: float = 0.8) -> pd.DataFrame:
    """
    Scan for potential crash signals using LPPL model over multiple windows.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    window_sizes : List[int]
        List of window sizes to check
    min_confidence : float
        Minimum confidence level to consider a signal valid
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with crash signal information
    """
    signals = []
    
    for window_size in window_sizes:
        # Skip if window is larger than available data
        if window_size >= len(prices):
            continue
        
        # Iterate through potential windows
        for start_idx in range(0, len(prices) - window_size, window_size // 4):
            end_idx = start_idx + window_size
            
            # Extract window
            window_prices = prices.iloc[start_idx:end_idx]
            
            # Fit LPPL model
            model = LPPLModel()
            try:
                params = model.fit(window_prices)
                
                # Check if parameters are in bubble regime
                # 0 < m < 1: positive feedback regime
                # omega within reasonable range
                # B < 0: positive bubble (price acceleration)
                is_bubble = (
                    0 < params.m < 1 and
                    6 < params.omega < 13 and
                    params.B < 0
                )
                
                # Calculate distance to critical time
                days_to_tc = params.tc - len(window_prices)
                
                # Only consider near-term signals
                if is_bubble and 0 < days_to_tc < 60:
                    # Estimated crash date
                    crash_date = window_prices.index[-1] + pd.Timedelta(days=int(days_to_tc))
                    
                    # Add to signals
                    signals.append({
                        'start_date': window_prices.index[0],
                        'end_date': window_prices.index[-1],
                        'window_size': window_size,
                        'm': params.m,
                        'omega': params.omega,
                        'B': params.B,
                        'days_to_crash': days_to_tc,
                        'estimated_crash_date': crash_date,
                        'confidence': min(1.0, 1.0 / days_to_tc * 30)  # Higher confidence for closer crashes
                    })
            except:
                # Skip failed fits
                continue
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(signals)
    
    # Filter by confidence
    if len(signals_df) > 0:
        signals_df = signals_df[signals_df['confidence'] >= min_confidence]
        
    return signals_df