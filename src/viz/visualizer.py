"""
Visualization module for equipment failure prediction.

This module provides comprehensive visualization tools for equipment failure
prediction models, including SHAP explanations, uncertainty quantification,
and interactive plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import shap

logger = logging.getLogger(__name__)


class EquipmentFailureVisualizer:
    """
    Comprehensive visualization tools for equipment failure prediction.
    
    This class provides various visualization methods for model analysis,
    SHAP explanations, and business insights.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.setup_plotting_style()
        
    def _get_default_config(self) -> Dict:
        """Get default visualization configuration."""
        return {
            "figure_size": (12, 8),
            "dpi": 100,
            "style": "seaborn-v0_8",
            "color_palette": "viridis",
            "save_format": "png",
            "save_dpi": 300
        }
    
    def setup_plotting_style(self) -> None:
        """Setup matplotlib and seaborn plotting styles."""
        plt.style.use(self.config["style"])
        sns.set_palette(self.config["color_palette"])
        
    def plot_prediction_scatter(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a scatter plot of predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.config["figure_size"])
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        
        # Add labels and title
        ax.set_xlabel('Actual Time to Failure (hours)')
        ax.set_ylabel('Predicted Time to Failure (hours)')
        ax.set_title(f'{model_name} - Predictions vs Actual\nR² = {r2:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        textstr = f'MAE: {mae:.2f} hours\nRMSE: {rmse:.2f} hours'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config["save_dpi"], bbox_inches='tight')
            logger.info(f"Prediction scatter plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create residual plots for model analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[0, 1].scatter(y_true, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Residual Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config["save_dpi"], bbox_inches='tight')
            logger.info(f"Residual plots saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self, 
        importance_df: pd.DataFrame,
        model_name: str = "Model",
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance from tree-based models.
        
        Args:
            importance_df: DataFrame with feature importance
            model_name: Name of the model
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=self.config["figure_size"])
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config["save_dpi"], bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_summary(
        self, 
        shap_values: np.ndarray, 
        X: pd.DataFrame,
        model_name: str = "Model",
        max_display: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            X: Feature matrix
            model_name: Name of the model
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.config["figure_size"])
        
        # Create SHAP summary plot
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        
        plt.title(f'{model_name} - SHAP Summary Plot')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config["save_dpi"], bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self, 
        results_df: pd.DataFrame,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create model comparison plots.
        
        Args:
            results_df: Model comparison DataFrame
            metrics: List of metrics to compare
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ["mae", "rmse", "r2", "mape"]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                
                # Create bar plot
                bars = ax.bar(results_df['model'], results_df[metric])
                
                # Customize plot
                ax.set_title(f'{metric.upper()} Comparison')
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.3f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config["save_dpi"], bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_business_impact(
        self, 
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot business impact metrics.
        
        Args:
            results_df: Model comparison DataFrame
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Maintenance cost savings
        axes[0, 0].bar(results_df['model'], results_df['maintenance_cost_savings'])
        axes[0, 0].set_title('Maintenance Cost Savings')
        axes[0, 0].set_ylabel('Cost Savings ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Downtime reduction
        axes[0, 1].bar(results_df['model'], results_df['downtime_reduction_hours'])
        axes[0, 1].set_title('Downtime Reduction')
        axes[0, 1].set_ylabel('Hours Saved')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Detection rate
        axes[1, 0].bar(results_df['model'], results_df['detection_rate'])
        axes[1, 0].set_title('Critical Failure Detection Rate')
        axes[1, 0].set_ylabel('Detection Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # False alarm rate
        axes[1, 1].bar(results_df['model'], results_df['false_alarm_rate'])
        axes[1, 1].set_title('False Alarm Rate')
        axes[1, 1].set_ylabel('False Alarm Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Business Impact Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config["save_dpi"], bbox_inches='tight')
            logger.info(f"Business impact plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(
        self, 
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            results_df: Model comparison DataFrame
            save_path: Path to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE Comparison', 'R² Comparison', 
                          'Business Value', 'Detection Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # MAE comparison
        fig.add_trace(
            go.Bar(x=results_df['model'], y=results_df['mae'], 
                  name='MAE', marker_color='lightblue'),
            row=1, col=1
        )
        
        # R² comparison
        fig.add_trace(
            go.Bar(x=results_df['model'], y=results_df['r2'], 
                  name='R²', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Business value
        fig.add_trace(
            go.Bar(x=results_df['model'], y=results_df['total_cost_savings'], 
                  name='Cost Savings', marker_color='orange'),
            row=2, col=1
        )
        
        # Detection rate
        fig.add_trace(
            go.Bar(x=results_df['model'], y=results_df['detection_rate'], 
                  name='Detection Rate', marker_color='red'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Equipment Failure Prediction - Model Comparison Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        
        fig.update_yaxes(title_text="MAE (hours)", row=1, col=1)
        fig.update_yaxes(title_text="R²", row=1, col=2)
        fig.update_yaxes(title_text="Cost Savings ($)", row=2, col=1)
        fig.update_yaxes(title_text="Detection Rate", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
