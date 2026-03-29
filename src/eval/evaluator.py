"""
Evaluation module for equipment failure prediction.

This module provides comprehensive evaluation metrics, business KPIs,
and model comparison tools for equipment failure prediction models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


class EquipmentFailureEvaluator:
    """
    Comprehensive evaluator for equipment failure prediction models.
    
    This class provides both standard ML metrics and business-specific
    KPIs for equipment failure prediction tasks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default evaluation configuration."""
        return {
            "maintenance_cost_per_hour": 100,  # Cost of maintenance per hour
            "downtime_cost_per_hour": 1000,    # Cost of downtime per hour
            "false_alarm_cost": 500,           # Cost of false alarm
            "critical_threshold": 50,          # Hours - critical failure threshold
            "high_threshold": 200,             # Hours - high risk threshold
            "medium_threshold": 500,           # Hours - medium risk threshold
        }
    
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            y_true: True time-to-failure values
            y_pred: Predicted time-to-failure values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Standard regression metrics
        metrics = {
            "model": model_name,
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "smape": self._symmetric_mape(y_true, y_pred),
        }
        
        # Business-specific metrics
        business_metrics = self._calculate_business_metrics(y_true, y_pred)
        metrics.update(business_metrics)
        
        # Risk classification metrics
        risk_metrics = self._calculate_risk_classification_metrics(y_true, y_pred)
        metrics.update(risk_metrics)
        
        return metrics
    
    def _symmetric_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate symmetric MAPE."""
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    def _calculate_business_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate business-specific metrics for equipment failure prediction."""
        
        # Maintenance cost savings (assuming proactive maintenance)
        maintenance_savings = self._calculate_maintenance_savings(y_true, y_pred)
        
        # Downtime reduction
        downtime_reduction = self._calculate_downtime_reduction(y_true, y_pred)
        
        # False alarm rate
        false_alarm_rate = self._calculate_false_alarm_rate(y_true, y_pred)
        
        # Detection rate
        detection_rate = self._calculate_detection_rate(y_true, y_pred)
        
        return {
            "maintenance_cost_savings": maintenance_savings,
            "downtime_reduction_hours": downtime_reduction,
            "false_alarm_rate": false_alarm_rate,
            "detection_rate": detection_rate,
            "total_cost_savings": maintenance_savings + downtime_reduction * self.config["downtime_cost_per_hour"]
        }
    
    def _calculate_maintenance_savings(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate maintenance cost savings from proactive maintenance."""
        # Assume proactive maintenance saves 20% of costs
        proactive_savings_rate = 0.2
        maintenance_cost_per_hour = self.config["maintenance_cost_per_hour"]
        
        # Calculate savings based on prediction accuracy
        prediction_error = np.abs(y_true - y_pred)
        avg_error = np.mean(prediction_error)
        
        # More accurate predictions lead to better planning and cost savings
        savings = (1 - avg_error / np.mean(y_true)) * proactive_savings_rate * maintenance_cost_per_hour * np.mean(y_true)
        
        return max(0, savings)
    
    def _calculate_downtime_reduction(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate downtime reduction from better failure prediction."""
        # Identify critical failures (TTF < threshold)
        critical_threshold = self.config["critical_threshold"]
        critical_true = y_true < critical_threshold
        critical_pred = y_pred < critical_threshold
        
        # True positives: correctly predicted critical failures
        true_positives = np.sum(critical_true & critical_pred)
        
        # False negatives: missed critical failures
        false_negatives = np.sum(critical_true & ~critical_pred)
        
        # Assume each correctly predicted critical failure saves 8 hours of downtime
        downtime_saved_per_prediction = 8
        
        # Calculate total downtime reduction
        downtime_reduction = true_positives * downtime_saved_per_prediction
        
        return downtime_reduction
    
    def _calculate_false_alarm_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false alarm rate for critical failure predictions."""
        critical_threshold = self.config["critical_threshold"]
        
        critical_pred = y_pred < critical_threshold
        critical_true = y_true < critical_threshold
        
        # False positives: predicted critical but not actually critical
        false_positives = np.sum(critical_pred & ~critical_true)
        total_predictions = len(y_pred)
        
        return false_positives / total_predictions if total_predictions > 0 else 0
    
    def _calculate_detection_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate detection rate for critical failures."""
        critical_threshold = self.config["critical_threshold"]
        
        critical_pred = y_pred < critical_threshold
        critical_true = y_true < critical_threshold
        
        # True positives: correctly predicted critical failures
        true_positives = np.sum(critical_pred & critical_true)
        total_critical = np.sum(critical_true)
        
        return true_positives / total_critical if total_critical > 0 else 0
    
    def _calculate_risk_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk classification metrics."""
        
        # Define risk categories
        def get_risk_category(ttf: float) -> str:
            if ttf < self.config["critical_threshold"]:
                return "critical"
            elif ttf < self.config["high_threshold"]:
                return "high"
            elif ttf < self.config["medium_threshold"]:
                return "medium"
            else:
                return "low"
        
        # Get risk categories
        true_risk = [get_risk_category(ttf) for ttf in y_true]
        pred_risk = [get_risk_category(ttf) for ttf in y_pred]
        
        # Calculate accuracy for risk classification
        risk_accuracy = np.mean([t == p for t, p in zip(true_risk, pred_risk)])
        
        # Calculate precision and recall for critical risk
        critical_true = np.array(true_risk) == "critical"
        critical_pred = np.array(pred_risk) == "critical"
        
        true_positives = np.sum(critical_true & critical_pred)
        false_positives = np.sum(critical_pred & ~critical_true)
        false_negatives = np.sum(critical_true & ~critical_pred)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "risk_classification_accuracy": risk_accuracy,
            "critical_precision": precision,
            "critical_recall": recall,
            "critical_f1": f1_score
        }
    
    def compare_models(
        self, 
        results: List[Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare multiple models and create a leaderboard.
        
        Args:
            results: List of evaluation results from different models
            
        Returns:
            DataFrame with model comparison
        """
        logger.info("Creating model comparison leaderboard")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by MAE (primary metric for regression)
        df = df.sort_values("mae")
        
        # Add ranking
        df["rank"] = range(1, len(df) + 1)
        
        # Reorder columns for better readability
        primary_metrics = ["rank", "model", "mae", "rmse", "r2", "mape"]
        business_metrics = ["maintenance_cost_savings", "downtime_reduction_hours", "total_cost_savings"]
        risk_metrics = ["risk_classification_accuracy", "critical_f1", "detection_rate", "false_alarm_rate"]
        
        column_order = primary_metrics + business_metrics + risk_metrics
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        
        df = df[available_columns + remaining_columns]
        
        return df
    
    def generate_evaluation_report(
        self, 
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results_df: Model comparison DataFrame
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("EQUIPMENT FAILURE PREDICTION - MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"Number of models evaluated: {len(results_df)}")
        report.append(f"Best model (by MAE): {results_df.iloc[0]['model']}")
        report.append(f"Best MAE: {results_df.iloc[0]['mae']:.2f} hours")
        report.append(f"Best R²: {results_df['r2'].max():.3f}")
        report.append("")
        
        # Top 3 models
        report.append("TOP 3 MODELS:")
        for i in range(min(3, len(results_df))):
            model = results_df.iloc[i]
            report.append(f"{i+1}. {model['model']}")
            report.append(f"   MAE: {model['mae']:.2f} hours")
            report.append(f"   RMSE: {model['rmse']:.2f} hours")
            report.append(f"   R²: {model['r2']:.3f}")
            report.append(f"   Business Value: ${model['total_cost_savings']:.0f}")
            report.append("")
        
        # Business impact analysis
        report.append("BUSINESS IMPACT ANALYSIS:")
        best_model = results_df.iloc[0]
        report.append(f"Maintenance Cost Savings: ${best_model['maintenance_cost_savings']:.0f}")
        report.append(f"Downtime Reduction: {best_model['downtime_reduction_hours']:.0f} hours")
        report.append(f"Total Cost Savings: ${best_model['total_cost_savings']:.0f}")
        report.append(f"Critical Failure Detection Rate: {best_model['detection_rate']:.1%}")
        report.append(f"False Alarm Rate: {best_model['false_alarm_rate']:.1%}")
        report.append("")
        
        # Risk classification performance
        report.append("RISK CLASSIFICATION PERFORMANCE:")
        report.append(f"Risk Classification Accuracy: {best_model['risk_classification_accuracy']:.1%}")
        report.append(f"Critical Risk F1-Score: {best_model['critical_f1']:.3f}")
        report.append("")
        
        # Detailed results table
        report.append("DETAILED RESULTS:")
        report.append(results_df.to_string(index=False, float_format='%.3f'))
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def calculate_confidence_intervals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with confidence intervals for key metrics
        """
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate standard error
        std_error = np.std(residuals)
        n = len(residuals)
        
        # Calculate confidence interval for MAE
        mae = np.mean(np.abs(residuals))
        mae_ci = (
            mae - 1.96 * std_error / np.sqrt(n),
            mae + 1.96 * std_error / np.sqrt(n)
        )
        
        # Calculate confidence interval for RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        rmse_ci = (
            rmse - 1.96 * std_error / np.sqrt(n),
            rmse + 1.96 * std_error / np.sqrt(n)
        )
        
        return {
            "mae_ci": mae_ci,
            "rmse_ci": rmse_ci,
            "confidence_level": confidence_level
        }
    
    def analyze_prediction_errors(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze prediction errors in detail.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with error analysis
        """
        errors = y_pred - y_true
        
        analysis = {
            "mean_error": np.mean(errors),
            "median_error": np.median(errors),
            "std_error": np.std(errors),
            "min_error": np.min(errors),
            "max_error": np.max(errors),
            "q25_error": np.percentile(errors, 25),
            "q75_error": np.percentile(errors, 75),
            "overestimation_rate": np.mean(errors > 0),
            "underestimation_rate": np.mean(errors < 0),
        }
        
        return analysis
