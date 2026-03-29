"""
Unit tests for Equipment Failure Prediction System.

This module contains comprehensive tests for all components
of the equipment failure prediction system.
"""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.models.equipment_failure_predictor import EquipmentFailurePredictor
from src.data.data_pipeline import EquipmentDataGenerator, DataPreprocessor
from src.eval.evaluator import EquipmentFailureEvaluator
from src.viz.visualizer import EquipmentFailureVisualizer


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return OmegaConf.create({
        "random_seed": 42,
        "data": {"n_samples": 100, "test_size": 0.25},
        "models": {
            "linear_regression": {"fit_intercept": True},
            "random_forest": {"n_estimators": 10, "max_depth": 5}
        }
    })


@pytest.fixture
def sample_data():
    """Sample equipment data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "equipment_id": [f"EQ_{i:04d}" for i in range(100)],
        "equipment_type": np.random.choice(["motor", "pump"], 100),
        "failure_mode": np.random.choice(["bearing_wear", "overheating"], 100),
        "temperature": np.random.uniform(60, 120, 100),
        "pressure": np.random.uniform(20, 50, 100),
        "vibration": np.random.uniform(0.1, 2.0, 100),
        "time_to_failure": np.random.uniform(10, 500, 100)
    })


class TestEquipmentFailurePredictor:
    """Test cases for EquipmentFailurePredictor."""
    
    def test_initialization(self, sample_config):
        """Test predictor initialization."""
        predictor = EquipmentFailurePredictor(sample_config)
        assert predictor.config == sample_config
        assert not predictor.is_fitted
        assert len(predictor.models) == 0
    
    def test_data_generation(self, sample_config):
        """Test synthetic data generation."""
        predictor = EquipmentFailurePredictor(sample_config)
        features, target = predictor.generate_synthetic_data(n_samples=50)
        
        assert len(features) == 50
        assert len(target) == 50
        assert len(features.columns) == 5  # Default number of features
        assert target.name == "time_to_failure"
        assert all(target > 0)  # TTF should be positive
    
    def test_model_training(self, sample_config, sample_data):
        """Test model training."""
        predictor = EquipmentFailurePredictor(sample_config)
        
        # Prepare data
        features = sample_data[["temperature", "pressure", "vibration"]]
        target = sample_data["time_to_failure"]
        
        # Train models
        predictor.fit_models(features, target)
        
        assert predictor.is_fitted
        assert len(predictor.models) > 0
        assert "linear_regression" in predictor.models
        assert "random_forest" in predictor.models
    
    def test_predictions(self, sample_config, sample_data):
        """Test model predictions."""
        predictor = EquipmentFailurePredictor(sample_config)
        
        # Prepare data
        features = sample_data[["temperature", "pressure", "vibration"]]
        target = sample_data["time_to_failure"]
        
        # Train models
        predictor.fit_models(features, target)
        
        # Make predictions
        test_features = features.head(10)
        predictions = predictor.predict(test_features)
        
        assert isinstance(predictions, dict)
        assert len(predictions) == len(predictor.models)
        
        for model_name, pred in predictions.items():
            assert len(pred) == len(test_features)
            assert all(pred > 0)  # Predictions should be positive
    
    def test_feature_importance(self, sample_config, sample_data):
        """Test feature importance extraction."""
        predictor = EquipmentFailurePredictor(sample_config)
        
        # Prepare data
        features = sample_data[["temperature", "pressure", "vibration"]]
        target = sample_data["time_to_failure"]
        
        # Train models
        predictor.fit_models(features, target)
        
        # Get feature importance
        importance = predictor.get_feature_importance("random_forest")
        
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == len(features.columns)


class TestEquipmentDataGenerator:
    """Test cases for EquipmentDataGenerator."""
    
    def test_initialization(self, sample_config):
        """Test generator initialization."""
        generator = EquipmentDataGenerator(sample_config)
        assert generator.config == sample_config
    
    def test_data_generation(self, sample_config):
        """Test equipment data generation."""
        generator = EquipmentDataGenerator(sample_config)
        
        df = generator.generate_equipment_data(n_samples=50)
        
        assert len(df) == 50
        assert "equipment_id" in df.columns
        assert "equipment_type" in df.columns
        assert "failure_mode" in df.columns
        assert "time_to_failure" in df.columns
        assert "failure_risk" in df.columns
        
        # Check data types
        assert df["time_to_failure"].dtype == float
        assert df["failure_risk"].dtype == object
    
    def test_sensor_readings(self, sample_config):
        """Test sensor reading generation."""
        generator = EquipmentDataGenerator(sample_config)
        
        sensor_data = generator._generate_sensor_readings("motor", "bearing_wear")
        
        assert isinstance(sensor_data, dict)
        assert "temperature" in sensor_data
        assert "pressure" in sensor_data
        assert "vibration" in sensor_data
        
        # Check reasonable ranges
        assert 0 < sensor_data["temperature"] < 200
        assert 0 < sensor_data["pressure"] < 100
        assert 0 < sensor_data["vibration"] < 10


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    def test_initialization(self, sample_config):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(sample_config)
        assert preprocessor.config == sample_config
    
    def test_preprocessing(self, sample_config, sample_data):
        """Test data preprocessing."""
        preprocessor = DataPreprocessor(sample_config)
        
        features, target, feature_names = preprocessor.preprocess_data(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert isinstance(feature_names, list)
        assert len(features) == len(sample_data)
        assert len(target) == len(sample_data)
        assert len(feature_names) == len(features.columns)
    
    def test_data_splitting(self, sample_config, sample_data):
        """Test data splitting."""
        preprocessor = DataPreprocessor(sample_config)
        
        features, target, _ = preprocessor.preprocess_data(sample_data)
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(features, target)
        
        # Check split sizes
        total_samples = len(features)
        expected_test_size = int(total_samples * sample_config.data.test_size)
        expected_val_size = int((total_samples - expected_test_size) * 0.2)
        expected_train_size = total_samples - expected_test_size - expected_val_size
        
        assert len(X_train) == expected_train_size
        assert len(X_val) == expected_val_size
        assert len(X_test) == expected_test_size
        
        # Check that splits don't overlap
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        assert len(train_indices.intersection(val_indices)) == 0
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(val_indices.intersection(test_indices)) == 0


class TestEquipmentFailureEvaluator:
    """Test cases for EquipmentFailureEvaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = EquipmentFailureEvaluator()
        assert evaluator.config is not None
    
    def test_evaluation(self):
        """Test model evaluation."""
        evaluator = EquipmentFailureEvaluator()
        
        # Create sample data
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = evaluator.evaluate_model(y_true, y_pred, "test_model")
        
        assert isinstance(metrics, dict)
        assert "model" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        
        # Check metric values
        assert metrics["model"] == "test_model"
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["r2"] <= 1
    
    def test_model_comparison(self):
        """Test model comparison."""
        evaluator = EquipmentFailureEvaluator()
        
        # Create sample results
        results = [
            {"model": "model1", "mae": 10.0, "rmse": 12.0, "r2": 0.8},
            {"model": "model2", "mae": 8.0, "rmse": 10.0, "r2": 0.85},
        ]
        
        comparison_df = evaluator.compare_models(results)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert "rank" in comparison_df.columns
        assert comparison_df.iloc[0]["model"] == "model2"  # Better MAE
        assert comparison_df.iloc[0]["rank"] == 1


class TestEquipmentFailureVisualizer:
    """Test cases for EquipmentFailureVisualizer."""
    
    def test_initialization(self):
        """Test visualizer initialization."""
        visualizer = EquipmentFailureVisualizer()
        assert visualizer.config is not None
    
    def test_config_defaults(self):
        """Test default configuration."""
        visualizer = EquipmentFailureVisualizer()
        config = visualizer._get_default_config()
        
        assert "figure_size" in config
        assert "dpi" in config
        assert "style" in config
        assert isinstance(config["figure_size"], tuple)
        assert isinstance(config["dpi"], int)


def test_integration(sample_config):
    """Integration test for the complete pipeline."""
    # Generate data
    predictor = EquipmentFailurePredictor(sample_config)
    features, target = predictor.generate_synthetic_data(n_samples=100)
    
    # Preprocess data
    preprocessor = DataPreprocessor(sample_config)
    processed_features, processed_target, feature_names = preprocessor.preprocess_data(
        pd.concat([features, target], axis=1)
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(
        processed_features, processed_target
    )
    
    # Train models
    predictor.fit_models(X_train, y_train)
    
    # Evaluate models
    evaluator = EquipmentFailureEvaluator()
    results = []
    
    for model_name in predictor.models.keys():
        y_pred = predictor.predict(X_test, model_name)
        metrics = evaluator.evaluate_model(y_test.values, y_pred, model_name)
        results.append(metrics)
    
    # Check results
    assert len(results) > 0
    assert all("mae" in result for result in results)
    assert all("r2" in result for result in results)
    
    # Test predictions
    test_input = X_test.head(1)
    predictions = predictor.predict(test_input)
    
    assert isinstance(predictions, dict)
    assert len(predictions) == len(predictor.models)


if __name__ == "__main__":
    pytest.main([__file__])
