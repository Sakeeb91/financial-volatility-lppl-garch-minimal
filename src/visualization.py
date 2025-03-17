import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

def set_plotting_style(style: str = 'seaborn-whitegrid', figsize: Tuple[int, int] = (12, 6)):
    """
    Set the global plotting style.
    
    Parameters:
    -----------
    style : str
        Matplotlib style
    figsize : Tuple[int, int]
        Default figure size
    """
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize

def plot_returns(returns: pd.Series, 
               title: str = 'Returns',
               style: str = None,
               figsize: Tuple[int, int] = (12, 6)):
    """
    Plot returns series.
    
    Parameters:
    -----------
    returns : pd.Series
        Series containing return data
    title : str
        Plot title
    style : str
        Matplotlib style to use
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")
        
    if style:
        plt.style.use(style)
        
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(returns.index, returns, color='blue', label='Returns')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax

def plot_volatility(volatility: Union[pd.Series, List[pd.Series]], 
                  labels: Optional[List[str]] = None,
                  title: str = 'Volatility',
                  figsize: Tuple[int, int] = (12, 6)):
    """
    Plot volatility series.
    
    Parameters:
    -----------
    volatility : Union[pd.Series, List[pd.Series]]
        Series or list of series containing volatility data
    labels : List[str], optional
        Labels for multiple volatility series
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if isinstance(volatility, pd.Series) and len(volatility) == 0:
        raise ValueError("Volatility series cannot be empty")
        
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(volatility, pd.Series):
        ax.plot(volatility.index, volatility, color='red', label='Volatility')
    else:
        colors = plt.cm.tab10.colors
        for i, vol in enumerate(volatility):
            label = labels[i] if labels and i < len(labels) else f'Series {i+1}'
            ax.plot(vol.index, vol, color=colors[i % len(colors)], label=label)
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax

def plot_returns_and_volatility(returns: pd.Series, 
                              volatility: Union[pd.Series, Dict[str, pd.Series]],
                              title: str = 'Returns and Volatility',
                              highlight_periods: Optional[Dict[str, Tuple[str, str]]] = None,
                              figsize: Tuple[int, int] = (12, 6)):
    """
    Plot returns and volatility together.
    
    Parameters:
    -----------
    returns : pd.Series
        Series containing return data
    volatility : Union[pd.Series, Dict[str, pd.Series]]
        Series containing volatility data or dictionary of multiple volatility series
    title : str
        Plot title
    highlight_periods : Dict[str, Tuple[str, str]], optional
        Dictionary mapping period names to (start_date, end_date) tuples to highlight
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot returns
    plt.plot(returns.index, returns, color='gray', alpha=0.5, label='Returns')
    
    # Plot volatility
    if isinstance(volatility, pd.Series):
        plt.plot(volatility.index, volatility, color='red', label='Volatility')
        plt.plot(volatility.index, -volatility, color='red')
    else:
        colors = plt.cm.tab10.colors
        for i, (name, vol_series) in enumerate(volatility.items()):
            color = colors[i % len(colors)]
            plt.plot(vol_series.index, vol_series, color=color, label=f'{name} Volatility')
    
    # Highlight periods if provided
    if highlight_periods:
        colors = ['red', 'orange', 'green', 'purple', 'brown']
        for i, (period_name, (start_date, end_date)) in enumerate(highlight_periods.items()):
            color = colors[i % len(colors)]
            plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date),
                      alpha=0.2, color=color, label=period_name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Returns / Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf(), plt.gca()

def plot_volatility_comparison(volatilities: Dict[str, pd.Series],
                             title: str = 'Volatility Model Comparison',
                             highlight_periods: Optional[Dict[str, Tuple[str, str]]] = None,
                             figsize: Tuple[int, int] = (12, 6)):
    """
    Plot multiple volatility series for comparison.
    
    Parameters:
    -----------
    volatilities : Dict[str, pd.Series]
        Dictionary mapping model names to volatility series
    title : str
        Plot title
    highlight_periods : Dict[str, Tuple[str, str]], optional
        Dictionary mapping period names to (start_date, end_date) tuples to highlight
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot volatilities
    colors = plt.cm.tab10.colors
    for i, (name, vol_series) in enumerate(volatilities.items()):
        color = colors[i % len(colors)]
        plt.plot(vol_series.index, vol_series, color=color, label=name)
    
    # Highlight periods if provided
    if highlight_periods:
        highlight_colors = ['red', 'orange', 'green', 'purple', 'brown']
        for i, (period_name, (start_date, end_date)) in enumerate(highlight_periods.items()):
            color = highlight_colors[i % len(highlight_colors)]
            plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date),
                      alpha=0.2, color=color, label=period_name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf(), plt.gca()

def plot_forecast_evaluation(eval_results: pd.DataFrame,
                           metric: str = 'rmse',
                           group_by: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)):
    """
    Plot evaluation metrics for different forecasting models.
    
    Parameters:
    -----------
    eval_results : pd.DataFrame
        DataFrame containing evaluation results
    metric : str
        Metric to plot ('rmse', 'mae', etc.)
    group_by : str, optional
        Column to group results by
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if group_by and group_by in eval_results.columns:
        # Grouped plot
        sns.barplot(x='model', y=metric, hue=group_by, data=eval_results)
    else:
        # Simple plot
        sns.barplot(x='model', y=metric, data=eval_results)
    
    plt.title(f'{metric.upper()} by Model')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf(), plt.gca()

def plot_rolling_window_results(results: pd.DataFrame,
                              metric_columns: List[str] = None,
                              model_names: List[str] = None,
                              figsize: Tuple[int, int] = (12, 8)):
    """
    Plot rolling window evaluation results.
    
    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame containing rolling window results
    metric_columns : List[str]
        List of column names for metrics to plot
    model_names : List[str]
        List of model names corresponding to metric columns
    figsize : Tuple[int, int]
        Figure size
    """
    if metric_columns is None:
        if isinstance(results, dict):
            # Handle dictionary of DataFrames
            fig, axes = plt.subplots(len(results), 1, figsize=figsize, sharex=True)
            
            for i, (window, df) in enumerate(results.items()):
                ax = axes[i] if len(results) > 1 else axes
                ax.plot(df.index, df['actual'], label='Actual', color='blue')
                ax.plot(df.index, df['forecast'], label='Forecast', color='red')
                ax.set_title(f'Window: {window}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig, axes
        else:
            # Default columns if not specified
            metric_columns = [col for col in results.columns if 'mse' in col.lower() or 'mae' in col.lower()]
            model_names = [col.split('_')[0].upper() for col in metric_columns]
    
    # Set date as index if not already
    if not isinstance(results.index, pd.DatetimeIndex):
        if 'start_date' in results.columns:
            results = results.set_index('start_date')
        elif 'train_end_date' in results.columns:
            results = results.set_index('train_end_date')
    
    fig = plt.figure(figsize=figsize)
    
    # Plot metrics
    for metric, model in zip(metric_columns, model_names):
        plt.plot(results.index, results[metric], label=f'{model}')
    
    plt.title('Rolling Window Forecast Performance')
    plt.xlabel('Date')
    plt.ylabel('Error Metric')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plot improvement if available
    if 'improvement_mse_pct' in results.columns:
        plt.figure(figsize=figsize)
        plt.bar(results.index, results['improvement_mse_pct'], color='green',
               alpha=0.7, label='Improvement (%)')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('LPPL-GARCH Improvement over Standard GARCH (%)')
        plt.xlabel('Date')
        plt.ylabel('Improvement (%)')
        plt.legend()
        plt.gca().yaxis.set_major_formatter(PercentFormatter())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    return fig, plt.gca() if len(metric_columns) > 0 else axes

def plot_lppl_fit(prices: pd.Series,
                fitted_values: Union[pd.Series, np.ndarray],
                tc: float = None,
                params: Dict[str, float] = None,
                ci_lower: Optional[Union[pd.Series, np.ndarray]] = None,
                ci_upper: Optional[Union[pd.Series, np.ndarray]] = None,
                figsize: Tuple[int, int] = (12, 6)):
    """
    Plot LPPL model fit to price data.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    fitted_values : Union[pd.Series, np.ndarray]
        Fitted LPPL values
    tc : float, optional
        Critical time parameter
    params : Dict[str, float], optional
        Dictionary of LPPL parameters
    ci_lower : Union[pd.Series, np.ndarray], optional
        Lower confidence interval
    ci_upper : Union[pd.Series, np.ndarray], optional
        Upper confidence interval
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Check if inputs have the same length
    if isinstance(fitted_values, pd.Series):
        if len(fitted_values) != len(prices):
            raise ValueError("Fitted values and prices must have the same length")
    elif len(fitted_values) != len(prices):
        raise ValueError("Fitted values and prices must have the same length")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original data
    ax.plot(prices.index, prices, label='Original Prices', color='blue')
    
    # Plot fitted values
    if isinstance(fitted_values, pd.Series):
        ax.plot(fitted_values.index, fitted_values, label='LPPL Fit', color='red', linestyle='--')
    else:
        valid_mask = np.isfinite(fitted_values)
        ax.plot(prices.index[valid_mask], fitted_values[valid_mask], label='LPPL Fit', color='red', linestyle='--')
    
    # Add confidence intervals if provided
    if ci_lower is not None and ci_upper is not None:
        if isinstance(ci_lower, pd.Series) and isinstance(ci_upper, pd.Series):
            ax.fill_between(ci_lower.index, ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')
        else:
            valid_mask = np.isfinite(ci_lower) & np.isfinite(ci_upper)
            ax.fill_between(prices.index[valid_mask], ci_lower[valid_mask], ci_upper[valid_mask], 
                          color='red', alpha=0.2, label='95% CI')
    
    # Add critical time marker if provided
    if tc is not None:
        if isinstance(prices.index, pd.DatetimeIndex):
            tc_date = prices.index[0] + pd.Timedelta(days=int(tc))
            ax.axvline(x=tc_date, color='green', linestyle='--', label='Critical Time (tc)')
        else:
            ax.axvline(x=tc, color='green', linestyle='--', label='Critical Time (tc)')
    
    # Add parameter annotations if provided
    if params is not None:
        params_text = '\n'.join([f"{k} = {v:.3f}" for k, v in params.items()])
        ax.annotate(params_text, xy=(0.02, 0.85), xycoords='axes fraction',
                  bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    ax.set_title('LPPL Model Fit')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax

def plot_model_comparison(models: Dict[str, pd.Series],
                        metrics: Optional[Dict[str, Dict[str, float]]] = None,
                        title: str = 'Model Comparison',
                        figsize: Tuple[int, int] = (12, 6)):
    """
    Plot comparison of multiple models.
    
    Parameters:
    -----------
    models : Dict[str, pd.Series]
        Dictionary mapping model names to series
    metrics : Dict[str, Dict[str, float]], optional
        Dictionary mapping model names to metrics
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot models
    colors = plt.cm.tab10.colors
    for i, (name, series) in enumerate(models.items()):
        color = colors[i % len(colors)]
        ax.plot(series.index, series, color=color, label=name)
    
    # Add metric annotations if provided
    if metrics is not None:
        y_pos = 0.95
        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            color = colors[i % len(colors)]
            metrics_text = ', '.join([f"{k}: {v:.4f}" for k, v in model_metrics.items()])
            ax.text(0.02, y_pos - i*0.05, f"{model_name}: {metrics_text}", 
                  transform=ax.transAxes, color=color,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax

def plot_crash_signals(returns: pd.Series,
                     volatility: pd.Series,
                     signals: pd.DataFrame,
                     window: int = 180,
                     figsize: Tuple[int, int] = (12, 6)):
    """
    Plot crash signals with returns and volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    volatility : pd.Series
        Volatility series
    signals : pd.DataFrame
        DataFrame containing crash signal information
    window : int
        Number of days around signals to plot
    figsize : Tuple[int, int]
        Figure size
    """
    if len(signals) == 0:
        print("No crash signals to plot.")
        return
    
    for i, signal in signals.iterrows():
        # Define plot window
        end_date = pd.to_datetime(signal['estimated_crash_date'])
        start_date = end_date - pd.Timedelta(days=window)
        plot_mask = (returns.index >= start_date) & (returns.index <= end_date + pd.Timedelta(days=window//2))
        
        plt.figure(figsize=figsize)
        
        # Plot returns
        plt.plot(returns[plot_mask].index, returns[plot_mask], 
                color='gray', alpha=0.5, label='Returns')
        
        # Plot volatility
        plt.plot(volatility[plot_mask].index, volatility[plot_mask], 
                color='red', label='Volatility')
        
        # Mark signal date
        signal_date = pd.to_datetime(signal['end_date'])
        plt.axvline(x=signal_date, color='blue', linestyle='--', 
                   label='Signal Date')
        
        # Mark estimated crash date
        plt.axvline(x=end_date, color='red', linestyle='--', 
                   label='Estimated Crash Date')
        
        # Add crash parameters
        params_text = (
            f"m = {signal['m']:.3f}\n"
            f"ω = {signal['omega']:.3f}\n"
            f"Days to crash: {signal['days_to_crash']:.1f}\n"
            f"Confidence: {signal['confidence']:.2f}"
        )
        
        plt.annotate(params_text, xy=(0.02, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.title(f'Crash Signal Analysis (Window: {signal["window_size"]} days)')
        plt.xlabel('Date')
        plt.ylabel('Returns / Volatility (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_crash_prediction_performance(prediction_results: pd.DataFrame,
                                   metrics: List[str] = ['warning_rate', 'earliest_warning'],
                                   figsize: Tuple[int, int] = (12, 6)):
    """
    Plot crash prediction performance metrics.
    
    Parameters:
    -----------
    prediction_results : pd.DataFrame
        DataFrame containing crash prediction results
    metrics : List[str]
        List of metrics to plot
    figsize : Tuple[int, int]
        Figure size
    """
    for metric in metrics:
        if metric in prediction_results.columns:
            plt.figure(figsize=figsize)
            sns.barplot(x='crash', y=metric, hue='model', data=prediction_results)
            plt.title(f'{metric.replace("_", " ").title()} by Model and Crash')
            plt.xlabel('Crash Event')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

def create_volatility_heatmap(volatilities: Dict[str, pd.Series],
                           freq: str = 'M',
                           figsize: Tuple[int, int] = (12, 8)):
    """
    Create a heatmap of volatilities by month.
    
    Parameters:
    -----------
    volatilities : Dict[str, pd.Series]
        Dictionary mapping model names to volatility series
    freq : str
        Frequency for resampling ('M' for month, 'Q' for quarter, etc.)
    figsize : Tuple[int, int]
        Figure size
    """
    # Create DataFrame with all volatilities
    vol_df = pd.DataFrame(volatilities)
    
    # Resample to specified frequency (monthly by default)
    resampled = vol_df.resample(freq).mean()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(resampled, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Volatility Heatmap')
    plt.xlabel('Model')
    plt.ylabel('Date')
    plt.tight_layout()
    plt.show()

def create_model_comparison_dashboard(returns: pd.Series,
                                    volatilities: Dict[str, pd.Series],
                                    eval_results: pd.DataFrame,
                                    crash_periods: Optional[Dict[str, Tuple[str, str]]] = None,
                                    figsize: Tuple[int, int] = (16, 12)):
    """
    Create a comprehensive dashboard for model comparison.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    volatilities : Dict[str, pd.Series]
        Dictionary mapping model names to volatility series
    eval_results : pd.DataFrame
        DataFrame containing evaluation results
    crash_periods : Dict[str, Tuple[str, str]], optional
        Dictionary mapping crash names to (start_date, end_date) tuples
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = plt.GridSpec(3, 2, figure=plt.gcf())
    
    # Volatility time series
    ax1 = plt.subplot(gs[0, :])
    colors = plt.cm.tab10.colors
    for i, (name, vol_series) in enumerate(volatilities.items()):
        color = colors[i % len(colors)]
        ax1.plot(vol_series.index, vol_series, color=color, label=name)
    
    # Add returns as background
    ax1_twin = ax1.twinx()
    ax1_twin.plot(returns.index, returns, color='gray', alpha=0.2, label='Returns')
    ax1_twin.set_ylabel('Returns (%)')
    
    # Highlight crash periods if provided
    if crash_periods:
        highlight_colors = ['red', 'orange', 'green', 'purple', 'brown']
        legend_elements = []
        
        for i, (period_name, (start_date, end_date)) in enumerate(crash_periods.items()):
            color = highlight_colors[i % len(highlight_colors)]
            ax1.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date),
                      alpha=0.2, color=color)
            
            # Add custom legend entry
            legend_elements.append(Patch(facecolor=color, alpha=0.2, label=period_name))
    
    ax1.set_title('Volatility Models Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volatility (%)')
    ax1.legend(loc='upper left')
    
    if crash_periods:
        ax1.legend(handles=ax1.get_lines() + legend_elements, loc='upper left')
    
    # Model performance metrics
    ax2 = plt.subplot(gs[1, 0])
    metrics = ['rmse', 'mae', 'mape'] if all(m in eval_results.columns for m in ['rmse', 'mae', 'mape']) else eval_results.columns[1:4]
    
    bar_positions = np.arange(len(metrics))
    bar_width = 0.8 / len(eval_results['model'].unique())
    
    for i, model in enumerate(eval_results['model'].unique()):
        model_data = eval_results[eval_results['model'] == model]
        values = [model_data[metric].values[0] for metric in metrics]
        ax2.bar(bar_positions + i * bar_width, values, width=bar_width, label=model)
    
    ax2.set_title('Forecast Error Metrics')
    ax2.set_xticks(bar_positions + bar_width * (len(eval_results['model'].unique()) - 1) / 2)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # Volatility heatmap
    ax3 = plt.subplot(gs[1, 1])
    
    # Create volatility difference series (LPPL-GARCH improvement over GARCH)
    if len(volatilities) >= 2:
        vol_series = list(volatilities.values())
        model_names = list(volatilities.keys())
        
        if 'GARCH' in model_names and any('LPPL' in name for name in model_names):
            garch_idx = model_names.index('GARCH')
            lppl_idx = next(i for i, name in enumerate(model_names) if 'LPPL' in name)
            
            # Calculate relative improvement
            garch_vol = vol_series[garch_idx]
            lppl_vol = vol_series[lppl_idx]
            
            # Make sure indices match
            common_idx = garch_vol.index.intersection(lppl_vol.index)
            improvement = (garch_vol[common_idx] - lppl_vol[common_idx]) / garch_vol[common_idx] * 100
            
            # Resample to monthly
            monthly_improvement = improvement.resample('M').mean()
            
            # Create heatmap data
            heatmap_data = monthly_improvement.values.reshape(-1, 1)
            
            # Plot heatmap
            im = ax3.imshow(heatmap_data.reshape(-1, 1).T, cmap='RdYlGn', aspect='auto')
            plt.colorbar(im, ax=ax3, label='Improvement (%)')
            
            # Set x-ticks to months
            months = [d.strftime('%Y-%m') for d in monthly_improvement.index]
            ax3.set_xticks(np.arange(len(months)))
            ax3.set_xticklabels(months, rotation=90)
            
            # Set y-ticks
            ax3.set_yticks([0])
            ax3.set_yticklabels(['LPPL-GARCH vs GARCH'])
            
            ax3.set_title('Monthly Volatility Improvement (%)')
    
    # Detailed analysis for one crash period
    if crash_periods:
        ax4 = plt.subplot(gs[2, :])
        
        # Use the first crash period
        crash_name = list(crash_periods.keys())[0]
        start_date, end_date = crash_periods[crash_name]
        
        # Extend window for better visualization
        plot_start = pd.to_datetime(start_date) - pd.Timedelta(days=30)
        plot_end = pd.to_datetime(end_date) + pd.Timedelta(days=30)
        plot_mask = (returns.index >= plot_start) & (returns.index <= plot_end)
        
        # Plot returns for this period
        ax4.plot(returns[plot_mask].index, returns[plot_mask], 
                color='gray', alpha=0.5, label='Returns')
        
        # Plot volatilities
        for i, (name, vol_series) in enumerate(volatilities.items()):
            mask = (vol_series.index >= plot_start) & (vol_series.index <= plot_end)
            ax4.plot(vol_series[mask].index, vol_series[mask], 
                    color=colors[i % len(colors)], label=f'{name}')
        
        # Highlight crash period
        ax4.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date),
                  alpha=0.2, color='red', label=f'{crash_name} Crash')
        
        ax4.set_title(f'Detailed View of {crash_name} Crash')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Returns / Volatility (%)')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()