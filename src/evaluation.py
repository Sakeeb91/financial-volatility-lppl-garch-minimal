import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf

def evaluate_volatility_forecast(actual: Union[pd.Series, np.ndarray], 
                               forecast: Union[pd.Series, np.ndarray],
                               model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate volatility forecast performance.
    
    Parameters:
    -----------
    actual : Union[pd.Series, np.ndarray]
        Actual volatility values (typically squared returns as proxy)
    forecast : Union[pd.Series, np.ndarray]
        Forecasted volatility values
    model_name : str
        Name of the model being evaluated
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    # Convert inputs to numpy arrays if needed
    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(forecast, pd.Series):
        forecast = forecast.values
    
    # Calculate metrics
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, forecast)
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mape = np.mean(np.abs((actual - forecast) / (actual + 1e-10))) * 100
    
    # Directional accuracy (for changes in volatility)
    if len(actual) > 1:
        actual_dir = np.sign(np.diff(np.concatenate(([actual[0]], actual))))
        forecast_dir = np.sign(np.diff(np.concatenate(([forecast[0]], forecast))))
        dir_accuracy = np.mean(actual_dir == forecast_dir) * 100
    else:
        dir_accuracy = np.nan
    
    # Mincer-Zarnowitz regression
    # volatility = a + b * forecast + error
    slope, intercept, r_value, p_value, std_err = stats.linregress(forecast, actual)
    r_squared = r_value**2
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'dir_accuracy': dir_accuracy,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'slope_p_value': p_value
    }

def compare_volatility_models(actual: Union[pd.Series, np.ndarray],
                            forecasts: Dict[str, Union[pd.Series, np.ndarray]],
                            metrics: List[str] = None,
                            plot: bool = True,
                            figsize: Tuple[int, int] = (12, 8)) -> pd.DataFrame:
    """
    Compare multiple volatility forecasting models.
    
    Parameters:
    -----------
    actual : Union[pd.Series, np.ndarray]
        Actual volatility values
    forecasts : Dict[str, Union[pd.Series, np.ndarray]]
        Dictionary mapping model names to forecasts
    metrics : List[str], optional
        List of metrics to include in comparison
    plot : bool
        Whether to plot comparison results
    figsize : Tuple[int, int]
        Figure size for plots
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing evaluation metrics for all models
    """
    # Default metrics if not specified
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'mape', 'dir_accuracy', 'r_squared']
    
    results = []
    
    # Evaluate each model
    for model_name, forecast_values in forecasts.items():
        eval_result = evaluate_volatility_forecast(actual, forecast_values, model_name)
        results.append(eval_result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot if requested
    if plot:
        # Create a bar chart for each metric
        for metric in metrics:
            if metric in results_df.columns:
                plt.figure(figsize=figsize)
                sns.barplot(x='model', y=metric, data=results_df)
                plt.title(f'{metric.upper()} by Model')
                plt.ylabel(metric.upper())
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
    
    return results_df

def forecast_accuracy_test(actual: Union[pd.Series, np.ndarray],
                        forecast1: Union[pd.Series, np.ndarray],
                        forecast2: Union[pd.Series, np.ndarray],
                        model1_name: str = "Model 1",
                        model2_name: str = "Model 2",
                        test_type: str = "dm",
                        loss: str = "MSE",
                        h: int = 1) -> Dict[str, Any]:
    """
    Perform statistical test to compare forecast accuracy of two models.
    
    Parameters:
    -----------
    actual : Union[pd.Series, np.ndarray]
        Actual values
    forecast1 : Union[pd.Series, np.ndarray]
        First model forecasts
    forecast2 : Union[pd.Series, np.ndarray]
        Second model forecasts
    model1_name : str
        Name of first model
    model2_name : str
        Name of second model
    test_type : str
        Test type ('dm' for Diebold-Mariano)
    loss : str
        Loss function ('MSE' or 'MAE')
    h : int
        Forecast horizon
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing test results
    """
    # Convert inputs to numpy arrays if needed
    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(forecast1, pd.Series):
        forecast1 = forecast1.values
    if isinstance(forecast2, pd.Series):
        forecast2 = forecast2.values
    
    if test_type.lower() == "dm":
        # Diebold-Mariano test
        # Calculate forecast errors
        e1 = actual - forecast1
        e2 = actual - forecast2
        
        # Calculate loss differential based on criterion
        if loss.upper() == "MSE":
            d = e1**2 - e2**2
        elif loss.upper() == "MAE":
            d = np.abs(e1) - np.abs(e2)
        else:
            raise ValueError("loss must be either 'MSE' or 'MAE'")
        
        # Calculate autocovariance for HAC estimator
        h = min(h, len(d)//2)
        gamma = np.zeros(h)
        for i in range(h):
            if len(d[i+1:]) > 0:
                gamma[i] = np.sum(d[i+1:] * d[:-i-1]) / len(d)
        
        # Calculate long-run variance
        v_d = gamma[0] + 2 * np.sum(gamma[1:])
        
        # Calculate DM statistic
        dm_stat = np.mean(d) / np.sqrt(v_d / len(d)) if v_d > 0 else np.nan
        
        # Calculate p-value
        p_value = 2 * stats.norm.cdf(-np.abs(dm_stat)) if not np.isnan(dm_stat) else np.nan
        
        # Interpret results
        if p_value < 0.05:
            if np.mean(d) < 0:
                conclusion = f"{model1_name} significantly outperforms {model2_name}"
            else:
                conclusion = f"{model2_name} significantly outperforms {model1_name}"
        else:
            conclusion = f"No significant difference between {model1_name} and {model2_name}"
        
        return {
            'test': 'Diebold-Mariano',
            'statistic': dm_stat,
            'p_value': p_value,
            'loss_function': loss,
            'model1': model1_name,
            'model2': model2_name,
            'conclusion': conclusion
        }
    else:
        raise ValueError(f"Unsupported test type: {test_type}")

def analyze_residuals(returns: pd.Series, 
                    volatility: pd.Series,
                    model_name: str = "Model",
                    figsize: Tuple[int, int] = (12, 10)) -> Dict[str, Any]:
    """
    Analyze standardized residuals from a volatility model.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    volatility : pd.Series
        Conditional volatility series
    model_name : str
        Name of the volatility model
    figsize : Tuple[int, int]
        Figure size for plots
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing residual analysis results
    """
    # Calculate standardized residuals
    std_residuals = returns / volatility
    
    # Test for normality
    k2, p_value = stats.normaltest(std_residuals.dropna())
    
    # Test for autocorrelation in squared residuals
    acf_values = acf(std_residuals.dropna()**2, nlags=20)
    ljung_box = stats.acorr_ljungbox(std_residuals.dropna()**2, lags=20)
    
    # Plot residual diagnostics
    plt.figure(figsize=figsize)
    
    # Standardized residuals
    plt.subplot(3, 2, 1)
    plt.plot(std_residuals.index, std_residuals)
    plt.title('Standardized Residuals')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Histogram with normal curve
    plt.subplot(3, 2, 2)
    sns.histplot(std_residuals.dropna(), kde=True, stat='density')
    x = np.linspace(std_residuals.min(), std_residuals.max(), 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
    plt.title('Histogram of Standardized Residuals')
    
    # QQ Plot
    plt.subplot(3, 2, 3)
    stats.probplot(std_residuals.dropna(), dist="norm", plot=plt)
    plt.title('QQ Plot')
    
    # ACF of residuals
    plt.subplot(3, 2, 4)
    plt.bar(range(1, len(acf_values)), acf_values[1:])
    plt.axhline(y=1.96/np.sqrt(len(std_residuals)), linestyle='--', color='r')
    plt.axhline(y=-1.96/np.sqrt(len(std_residuals)), linestyle='--', color='r')
    plt.title('ACF of Squared Standardized Residuals')
    plt.xlabel('Lag')
    
    # Volatility vs Absolute Returns
    plt.subplot(3, 2, 5)
    plt.scatter(volatility, abs(returns), alpha=0.5)
    plt.plot([0, max(volatility)], [0, max(volatility)], 'r--')
    plt.title('Volatility vs Absolute Returns')
    plt.xlabel('Predicted Volatility')
    plt.ylabel('Absolute Returns')
    
    # Residuals over time with crash periods highlighted
    plt.subplot(3, 2, 6)
    plt.plot(std_residuals.index, std_residuals**2)
    plt.title('Squared Standardized Residuals')
    
    plt.tight_layout()
    plt.suptitle(f'Residual Diagnostics for {model_name}', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return {
        'model': model_name,
        'normality_test': {
            'statistic': k2,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        },
        'autocorrelation_test': {
            'ljung_box_statistic': ljung_box[0],
            'ljung_box_p_value': ljung_box[1],
            'no_autocorrelation': all(p > 0.05 for p in ljung_box[1])
        }
    }

def evaluate_crash_prediction(returns: pd.Series,
                           volatility_models: Dict[str, pd.Series],
                           crash_periods: Dict[str, Tuple[str, str]],
                           warning_threshold: float = 1.5,
                           lead_time: int = 10,
                           plot: bool = True,
                           figsize: Tuple[int, int] = (12, 8)) -> pd.DataFrame:
    """
    Evaluate model performance for crash prediction.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    volatility_models : Dict[str, pd.Series]
        Dictionary mapping model names to volatility series
    crash_periods : Dict[str, Tuple[str, str]]
        Dictionary mapping crash names to (start_date, end_date) tuples
    warning_threshold : float
        Volatility threshold multiplier for issuing warnings
    lead_time : int
        Number of days before crash to evaluate warnings
    plot : bool
        Whether to plot crash prediction results
    figsize : Tuple[int, int]
        Figure size for plots
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with crash prediction performance metrics
    """
    results = []
    
    for crash_name, (start_date, end_date) in crash_periods.items():
        # Define pre-crash period
        pre_crash_start = pd.to_datetime(start_date) - pd.Timedelta(days=lead_time)
        pre_crash_end = pd.to_datetime(start_date)
        
        # Create mask for pre-crash period
        pre_crash_mask = (returns.index >= pre_crash_start) & (returns.index < pre_crash_end)
        
        for model_name, volatility in volatility_models.items():
            # Calculate baseline volatility (median over previous 6 months)
            baseline_end = pre_crash_start
            baseline_start = baseline_end - pd.Timedelta(days=180)
            baseline_mask = (volatility.index >= baseline_start) & (volatility.index < baseline_end)
            baseline_vol = volatility[baseline_mask].median()
            
            # Check for warnings in pre-crash period
            pre_crash_vol = volatility[pre_crash_mask]
            warnings_issued = (pre_crash_vol > baseline_vol * warning_threshold).sum()
            warning_days = len(pre_crash_vol)
            
            # Calculate warning rate
            warning_rate = warnings_issued / warning_days if warning_days > 0 else 0
            
            # Calculate earliest warning
            if warnings_issued > 0:
                warning_indices = np.where(pre_crash_vol > baseline_vol * warning_threshold)[0]
                earliest_warning = warning_days - min(warning_indices)
            else:
                earliest_warning = 0
            
            # Store results
            results.append({
                'crash': crash_name,
                'model': model_name,
                'warnings_issued': warnings_issued,
                'warning_days': warning_days,
                'warning_rate': warning_rate,
                'earliest_warning': earliest_warning,
                'baseline_volatility': baseline_vol,
                'threshold': baseline_vol * warning_threshold
            })
    
    results_df = pd.DataFrame(results)
    
    # Plot if requested
    if plot:
        # Plot warning rates by model and crash
        plt.figure(figsize=figsize)
        sns.barplot(x='crash', y='warning_rate', hue='model', data=results_df)
        plt.title('Warning Rate by Model and Crash')
        plt.ylabel('Warning Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot earliest warnings
        plt.figure(figsize=figsize)
        sns.barplot(x='crash', y='earliest_warning', hue='model', data=results_df)
        plt.title('Days Before Crash of Earliest Warning')
        plt.ylabel('Days')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot volatility around crash periods
        for crash_name, (start_date, end_date) in crash_periods.items():
            plt.figure(figsize=figsize)
            
            # Define plot period (3 months before to 1 month after)
            plot_start = pd.to_datetime(start_date) - pd.Timedelta(days=90)
            plot_end = pd.to_datetime(end_date) + pd.Timedelta(days=30)
            plot_mask = (returns.index >= plot_start) & (returns.index <= plot_end)
            
            # Plot returns
            plt.plot(returns[plot_mask].index, returns[plot_mask], 
                    color='blue', alpha=0.5, label='Returns')
            
            # Plot volatilities
            for model_name, volatility in volatility_models.items():
                plt.plot(volatility[plot_mask].index, volatility[plot_mask], 
                        label=f'{model_name} Volatility')
            
            # Add crash period shading
            plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), 
                       alpha=0.2, color='red', label=f'{crash_name} Crash')
            
            # Add pre-crash period shading
            pre_crash_start = pd.to_datetime(start_date) - pd.Timedelta(days=lead_time)
            plt.axvspan(pre_crash_start, pd.to_datetime(start_date), 
                       alpha=0.2, color='yellow', label='Pre-Crash Warning Period')
            
            plt.title(f'Volatility Around {crash_name} Crash')
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    return results_df