#!/usr/bin/env python3
"""
Main training script for Equipment Failure Prediction System.

This script demonstrates the complete workflow from data generation
to model training and evaluation.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --n-samples 2000 --models xgboost lightgbm
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.equipment_failure_predictor import EquipmentFailurePredictor
from data.data_pipeline import EquipmentDataGenerator, DataPreprocessor
from eval.evaluator import EquipmentFailureEvaluator
from viz.visualizer import EquipmentFailureVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Equipment Failure Prediction Models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to train"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained models"
    )
    
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> DictConfig:
    """Load configuration from file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return OmegaConf.create({
            "random_seed": 42,
            "data": {"n_samples": 1000, "test_size": 0.25},
            "models": {
                "linear_regression": {"fit_intercept": True},
                "random_forest": {"n_estimators": 100, "max_depth": 10},
                "xgboost": {"n_estimators": 100, "max_depth": 6},
                "lightgbm": {"n_estimators": 100, "max_depth": 6}
            }
        })
    
    return OmegaConf.load(config_file)


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting Equipment Failure Prediction Training")
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.n_samples:
        config.data.n_samples = args.n_samples
    
    if args.models:
        # Filter config to only include selected models
        selected_models = {model: config.models[model] for model in args.models if model in config.models}
        config.models = selected_models
    
    logger.info(f"Configuration: {config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate data
    logger.info("Step 1: Generating synthetic equipment data")
    generator = EquipmentDataGenerator(config)
    
    df = generator.generate_equipment_data(
        n_samples=config.data.n_samples,
        equipment_types=["motor", "pump", "compressor", "generator"],
        failure_modes=["bearing_wear", "overheating", "vibration_fatigue", "electrical_fault"]
    )
    
    # Save raw data
    data_path = output_dir / "raw_data.csv"
    df.to_csv(data_path, index=False)
    logger.info(f"Raw data saved to {data_path}")
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data")
    preprocessor = DataPreprocessor(config)
    
    features, target, feature_names = preprocessor.preprocess_data(df)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(features, target)
    
    # Save processed data
    processed_data_path = output_dir / "processed_data.csv"
    processed_df = features.copy()
    processed_df['time_to_failure'] = target
    processed_df.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    # Step 3: Train models
    logger.info("Step 3: Training models")
    predictor = EquipmentFailurePredictor(config)
    
    predictor.fit_models(X_train, y_train)
    logger.info(f"Successfully trained {len(predictor.models)} models")
    
    # Step 4: Evaluate models
    logger.info("Step 4: Evaluating models")
    evaluator = EquipmentFailureEvaluator()
    
    results = []
    for model_name in predictor.models.keys():
        y_pred = predictor.predict(X_test, model_name)
        metrics = evaluator.evaluate_model(y_test.values, y_pred, model_name)
        results.append(metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = evaluator.compare_models(results)
    
    # Save results
    results_path = output_dir / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Model results saved to {results_path}")
    
    # Generate evaluation report
    report_path = output_dir / "evaluation_report.txt"
    report = evaluator.generate_evaluation_report(results_df, str(report_path))
    logger.info(f"Evaluation report saved to {report_path}")
    
    # Print results
    print("\n" + "="*60)
    print("EQUIPMENT FAILURE PREDICTION - TRAINING RESULTS")
    print("="*60)
    print(f"\nDataset: {len(df)} samples, {len(feature_names)} features")
    print(f"Models trained: {len(predictor.models)}")
    print(f"Test samples: {len(X_test)}")
    
    print("\nModel Performance:")
    print(results_df[['model', 'mae', 'rmse', 'r2', 'mape']].round(3).to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['model']}")
    print(f"MAE: {best_model['mae']:.2f} hours")
    print(f"R²: {best_model['r2']:.3f}")
    print(f"Business Value: ${best_model['total_cost_savings']:.0f}")
    
    # Step 5: Generate visualizations
    if args.generate_plots:
        logger.info("Step 5: Generating visualizations")
        visualizer = EquipmentFailureVisualizer()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot for best model
        best_model_name = best_model['model']
        y_pred_best = predictor.predict(X_test, best_model_name)
        
        # Prediction scatter plot
        visualizer.plot_prediction_scatter(
            y_test.values, y_pred_best, best_model_name,
            save_path=str(plots_dir / "prediction_scatter.png")
        )
        
        # Residual plots
        visualizer.plot_residuals(
            y_test.values, y_pred_best, best_model_name,
            save_path=str(plots_dir / "residuals.png")
        )
        
        # Model comparison
        visualizer.plot_model_comparison(
            results_df,
            save_path=str(plots_dir / "model_comparison.png")
        )
        
        # Business impact
        visualizer.plot_business_impact(
            results_df,
            save_path=str(plots_dir / "business_impact.png")
        )
        
        # Feature importance (if available)
        if best_model_name in predictor.models and hasattr(predictor.models[best_model_name], 'feature_importances_'):
            importance_df = predictor.get_feature_importance(best_model_name)
            visualizer.plot_feature_importance(
                importance_df, best_model_name,
                save_path=str(plots_dir / "feature_importance.png")
            )
        
        logger.info(f"Plots saved to {plots_dir}")
    
    # Step 6: Save models
    if args.save_models:
        logger.info("Step 6: Saving trained models")
        models_dir = output_dir / "models"
        predictor.save_models(models_dir)
        logger.info(f"Models saved to {models_dir}")
    
    # Step 7: Example predictions
    logger.info("Step 7: Generating example predictions")
    
    # Create example inputs
    example_inputs = [
        {"temperature": 85, "pressure": 40, "vibration": 0.8, "current": 8, "voltage": 225},
        {"temperature": 95, "pressure": 45, "vibration": 1.2, "current": 12, "voltage": 230},
        {"temperature": 75, "pressure": 35, "vibration": 0.5, "current": 6, "voltage": 220},
    ]
    
    print("\nExample Predictions:")
    print("-" * 40)
    
    for i, example in enumerate(example_inputs):
        example_df = pd.DataFrame([example])
        
        # Make predictions with all models
        predictions = predictor.predict(example_df)
        
        print(f"\nExample {i+1}: {example}")
        for model_name, pred in predictions.items():
            ttf = pred[0]
            risk = "Critical" if ttf < 50 else "High" if ttf < 200 else "Medium" if ttf < 500 else "Low"
            print(f"  {model_name}: {ttf:.1f} hours ({risk} risk)")
    
    logger.info("Training completed successfully!")
    print(f"\nAll outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
