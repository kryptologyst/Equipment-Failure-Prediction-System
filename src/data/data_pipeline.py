"""
Data pipeline for equipment failure prediction.

This module handles data generation, preprocessing, and feature engineering
for equipment failure prediction tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class EquipmentDataGenerator:
    """
    Generate synthetic equipment failure data for research and education.
    
    This class creates realistic sensor data that correlates with equipment
    failure patterns, including multiple sensor types and failure modes.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_names = [
            "temperature", "pressure", "vibration", "current", "voltage"
        ]
        
    def generate_equipment_data(
        self, 
        n_samples: int = 1000,
        equipment_types: Optional[List[str]] = None,
        failure_modes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive equipment failure dataset.
        
        Args:
            n_samples: Number of samples to generate
            equipment_types: List of equipment types
            failure_modes: List of failure modes
            
        Returns:
            DataFrame with equipment data
        """
        logger.info(f"Generating {n_samples} equipment samples")
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        # Default equipment types and failure modes
        if equipment_types is None:
            equipment_types = ["motor", "pump", "compressor", "generator", "turbine"]
        
        if failure_modes is None:
            failure_modes = ["bearing_wear", "overheating", "vibration_fatigue", "electrical_fault"]
        
        # Generate base data
        data = []
        
        for i in range(n_samples):
            # Random equipment type and failure mode
            equipment_type = np.random.choice(equipment_types)
            failure_mode = np.random.choice(failure_modes)
            
            # Generate sensor readings based on equipment type and failure mode
            sensor_data = self._generate_sensor_readings(equipment_type, failure_mode)
            
            # Calculate time to failure based on sensor readings and failure mode
            ttf = self._calculate_time_to_failure(sensor_data, failure_mode)
            
            # Create sample
            sample = {
                "equipment_id": f"EQ_{i:04d}",
                "equipment_type": equipment_type,
                "failure_mode": failure_mode,
                "age_hours": np.random.uniform(1000, 50000),  # Equipment age
                "operating_hours": np.random.uniform(5000, 8000),  # Hours per year
                "maintenance_frequency": np.random.choice(["daily", "weekly", "monthly"]),
                **sensor_data,
                "time_to_failure": ttf,
                "failure_risk": self._calculate_failure_risk(ttf)
            }
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Generated dataset with {df.shape[0]} samples and {df.shape[1]} features")
        logger.info(f"Equipment types: {df['equipment_type'].value_counts().to_dict()}")
        logger.info(f"Failure modes: {df['failure_mode'].value_counts().to_dict()}")
        
        return df
    
    def _generate_sensor_readings(self, equipment_type: str, failure_mode: str) -> Dict[str, float]:
        """Generate realistic sensor readings based on equipment type and failure mode."""
        
        # Base sensor ranges by equipment type
        sensor_ranges = {
            "motor": {"temperature": (60, 120), "pressure": (20, 50), "vibration": (0.1, 2.0), 
                     "current": (5, 15), "voltage": (200, 250)},
            "pump": {"temperature": (50, 100), "pressure": (30, 80), "vibration": (0.2, 3.0), 
                    "current": (8, 20), "voltage": (200, 250)},
            "compressor": {"temperature": (70, 130), "pressure": (40, 100), "vibration": (0.3, 4.0), 
                         "current": (10, 25), "voltage": (200, 250)},
            "generator": {"temperature": (80, 140), "pressure": (25, 60), "vibration": (0.1, 1.5), 
                        "current": (15, 30), "voltage": (220, 260)},
            "turbine": {"temperature": (90, 150), "pressure": (50, 120), "vibration": (0.2, 2.5), 
                       "current": (12, 28), "voltage": (200, 250)}
        }
        
        ranges = sensor_ranges.get(equipment_type, sensor_ranges["motor"])
        
        # Generate base readings
        readings = {}
        for sensor, (min_val, max_val) in ranges.items():
            readings[sensor] = np.random.uniform(min_val, max_val)
        
        # Modify readings based on failure mode
        failure_modifications = {
            "bearing_wear": {"vibration": 1.5, "temperature": 1.2},
            "overheating": {"temperature": 2.0, "current": 1.3},
            "vibration_fatigue": {"vibration": 2.0, "pressure": 1.1},
            "electrical_fault": {"current": 1.8, "voltage": 0.8, "temperature": 1.3}
        }
        
        modifications = failure_modifications.get(failure_mode, {})
        for sensor, multiplier in modifications.items():
            if sensor in readings:
                readings[sensor] *= multiplier
        
        # Add noise
        for sensor in readings:
            noise = np.random.normal(0, 0.05 * readings[sensor])
            readings[sensor] += noise
        
        return readings
    
    def _calculate_time_to_failure(self, sensor_data: Dict[str, float], failure_mode: str) -> float:
        """Calculate time to failure based on sensor readings and failure mode."""
        
        # Base time to failure (hours)
        base_ttf = 1000
        
        # Failure mode specific calculations
        if failure_mode == "bearing_wear":
            ttf_reduction = (
                sensor_data["vibration"] * 50 +
                sensor_data["temperature"] * 2 +
                sensor_data["pressure"] * 1
            )
        elif failure_mode == "overheating":
            ttf_reduction = (
                sensor_data["temperature"] * 5 +
                sensor_data["current"] * 10 +
                sensor_data["vibration"] * 20
            )
        elif failure_mode == "vibration_fatigue":
            ttf_reduction = (
                sensor_data["vibration"] * 100 +
                sensor_data["pressure"] * 2 +
                sensor_data["temperature"] * 1
            )
        elif failure_mode == "electrical_fault":
            ttf_reduction = (
                sensor_data["current"] * 15 +
                sensor_data["voltage"] * 0.5 +
                sensor_data["temperature"] * 3
            )
        else:
            ttf_reduction = sum(sensor_data.values()) * 2
        
        # Calculate final TTF
        ttf = base_ttf - ttf_reduction
        
        # Ensure positive TTF
        ttf = max(ttf, 1)
        
        # Add some randomness
        ttf *= np.random.uniform(0.8, 1.2)
        
        return ttf
    
    def _calculate_failure_risk(self, ttf: float) -> str:
        """Calculate failure risk category based on time to failure."""
        if ttf < 50:
            return "critical"
        elif ttf < 200:
            return "high"
        elif ttf < 500:
            return "medium"
        else:
            return "low"
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset."""
        
        # Temperature-pressure ratio
        df["temp_pressure_ratio"] = df["temperature"] / df["pressure"]
        
        # Vibration severity
        df["vibration_severity"] = pd.cut(
            df["vibration"], 
            bins=[0, 1, 2, 3, float('inf')], 
            labels=["low", "medium", "high", "critical"]
        )
        
        # Temperature trend (simulated)
        df["temp_trend"] = np.random.choice(["increasing", "stable", "decreasing"], len(df))
        
        # Equipment utilization
        df["utilization"] = df["operating_hours"] / 8760  # Hours per year
        
        # Maintenance effectiveness (simulated)
        maintenance_effectiveness = {
            "daily": 0.9,
            "weekly": 0.7,
            "monthly": 0.5
        }
        df["maintenance_effectiveness"] = df["maintenance_frequency"].map(maintenance_effectiveness)
        
        return df


class DataPreprocessor:
    """
    Preprocess equipment failure data for modeling.
    
    This class handles data cleaning, feature engineering, and preparation
    for machine learning models.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scalers = {}
        self.feature_names = []
        
    def preprocess_data(
        self, 
        df: pd.DataFrame,
        target_column: str = "time_to_failure"
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Preprocess the equipment data for modeling.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (features_df, target_series, feature_names)
        """
        logger.info("Preprocessing equipment data")
        
        # Separate features and target
        target = df[target_column]
        features_df = df.drop(columns=[target_column])
        
        # Handle categorical variables
        features_df = self._encode_categorical_features(features_df)
        
        # Select numerical features for modeling
        numerical_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        features_df = features_df[numerical_features]
        
        # Store feature names
        self.feature_names = numerical_features
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Feature engineering
        if self.config.feature_engineering.create_interactions:
            features_df = self._create_interaction_features(features_df)
        
        if self.config.feature_engineering.create_polynomials:
            features_df = self._create_polynomial_features(features_df)
        
        logger.info(f"Preprocessed data: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        
        return features_df, target, self.feature_names
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        # One-hot encode categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found missing values: {df.isnull().sum().sum()}")
            # Fill missing values with median for numerical columns
            df = df.fillna(df.median())
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df_interactions = df.copy()
        
        # Create specified interaction features
        interaction_pairs = self.config.feature_engineering.interaction_features
        
        for pair in interaction_pairs:
            if len(pair) == 2 and all(feat in df.columns for feat in pair):
                interaction_name = f"{pair[0]}_x_{pair[1]}"
                df_interactions[interaction_name] = df[pair[0]] * df[pair[1]]
        
        return df_interactions
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""
        poly = PolynomialFeatures(
            degree=self.config.feature_engineering.polynomial_degree,
            include_bias=False,
            interaction_only=False
        )
        
        # Fit and transform
        poly_features = poly.fit_transform(df)
        poly_feature_names = poly.get_feature_names_out(df.columns)
        
        # Create new DataFrame
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        
        return df_poly
    
    def split_data(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            features: Feature matrix
            target: Target variable
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Splitting data into train/validation/test sets")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target,
            test_size=self.config.data.test_size,
            random_state=self.config.random_seed
        )
        
        # Second split: train vs val
        val_size = self.config.data.validation_size / (1 - self.config.data.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.random_seed
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_val: Optional[pd.DataFrame] = None, 
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled features
        """
        logger.info("Scaling features")
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        self.scalers["standard"] = scaler
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, X_val_scaled, X_test_scaled


def create_sample_dataset(config: DictConfig) -> pd.DataFrame:
    """
    Create a sample equipment failure dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Sample dataset
    """
    generator = EquipmentDataGenerator(config)
    
    # Generate data
    df = generator.generate_equipment_data(
        n_samples=config.data.n_samples,
        equipment_types=["motor", "pump", "compressor", "generator"],
        failure_modes=["bearing_wear", "overheating", "vibration_fatigue", "electrical_fault"]
    )
    
    return df
