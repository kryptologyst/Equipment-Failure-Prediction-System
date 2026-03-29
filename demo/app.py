"""
Streamlit demo for Equipment Failure Prediction System.

This interactive demo showcases the equipment failure prediction capabilities
with real-time predictions, model comparison, and business impact analysis.

DISCLAIMER: This is an experimental research/educational project.
DO NOT use for automated maintenance decisions without human review.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.equipment_failure_predictor import EquipmentFailurePredictor
from src.data.data_pipeline import EquipmentDataGenerator, DataPreprocessor
from src.eval.evaluator import EquipmentFailureEvaluator
from src.viz.visualizer import EquipmentFailureVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Equipment Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_config() -> Dict:
    """Load configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    else:
        # Return default config
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


@st.cache_data
def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample equipment data."""
    config = load_config()
    generator = EquipmentDataGenerator(config)
    return generator.generate_equipment_data(n_samples=n_samples)


@st.cache_resource
def load_predictor() -> EquipmentFailurePredictor:
    """Load and cache the predictor."""
    config = load_config()
    return EquipmentFailurePredictor(config)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">⚙️ Equipment Failure Prediction System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ DISCLAIMER:</strong> This is an experimental research/educational project. 
        <strong>DO NOT use for automated maintenance decisions without human review.</strong>
        All predictions should be validated by qualified maintenance professionals.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000)
    equipment_types = st.sidebar.multiselect(
        "Equipment Types",
        ["motor", "pump", "compressor", "generator", "turbine"],
        default=["motor", "pump", "compressor", "generator"]
    )
    failure_modes = st.sidebar.multiselect(
        "Failure Modes",
        ["bearing_wear", "overheating", "vibration_fatigue", "electrical_fault"],
        default=["bearing_wear", "overheating", "vibration_fatigue", "electrical_fault"]
    )
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    selected_models = st.sidebar.multiselect(
        "Models to Train",
        ["linear_regression", "random_forest", "xgboost", "lightgbm"],
        default=["linear_regression", "random_forest", "xgboost", "lightgbm"]
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Predictions", 
        "💼 Business Impact", 
        "🔍 Model Analysis"
    ])
    
    with tab1:
        show_data_overview(n_samples, equipment_types, failure_modes)
    
    with tab2:
        show_model_training(selected_models, n_samples)
    
    with tab3:
        show_predictions()
    
    with tab4:
        show_business_impact()
    
    with tab5:
        show_model_analysis()


def show_data_overview(n_samples: int, equipment_types: List[str], failure_modes: List[str]):
    """Show data overview tab."""
    st.header("📊 Data Overview")
    
    # Generate data
    with st.spinner("Generating sample data..."):
        config = load_config()
        generator = EquipmentDataGenerator(config)
        df = generator.generate_equipment_data(
            n_samples=n_samples,
            equipment_types=equipment_types,
            failure_modes=failure_modes
        )
    
    # Store in session state
    st.session_state['data'] = df
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        st.metric("Features", len(df.columns) - 1)  # Exclude target
    
    with col3:
        st.metric("Equipment Types", df['equipment_type'].nunique())
    
    with col4:
        st.metric("Failure Modes", df['failure_mode'].nunique())
    
    # Data distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Equipment type distribution
        fig = px.pie(
            df, 
            names='equipment_type', 
            title='Equipment Type Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Failure mode distribution
        fig = px.pie(
            df, 
            names='failure_mode', 
            title='Failure Mode Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sensor readings distribution
    st.subheader("Sensor Readings Distribution")
    
    sensor_cols = ['temperature', 'pressure', 'vibration', 'current', 'voltage']
    available_sensors = [col for col in sensor_cols if col in df.columns]
    
    if available_sensors:
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=available_sensors[:6],
            specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)]
        )
        
        for i, sensor in enumerate(available_sensors[:6]):
            row = i // 3 + 1
            col = i % 3 + 1
            
            fig.add_trace(
                go.Histogram(x=df[sensor], name=sensor, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Sensor Readings Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time to failure distribution
    st.subheader("Time to Failure Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x='time_to_failure', 
            nbins=50,
            title='Time to Failure Distribution',
            labels={'time_to_failure': 'Time to Failure (hours)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution
        risk_counts = df['failure_risk'].value_counts()
        fig = px.bar(
            x=risk_counts.index, 
            y=risk_counts.values,
            title='Failure Risk Distribution',
            labels={'x': 'Risk Level', 'y': 'Count'},
            color=risk_counts.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)


def show_model_training(selected_models: List[str], n_samples: int):
    """Show model training tab."""
    st.header("🤖 Model Training")
    
    if 'data' not in st.session_state:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    df = st.session_state['data']
    
    # Prepare data
    with st.spinner("Preparing data..."):
        config = load_config()
        preprocessor = DataPreprocessor(config)
        
        features, target, feature_names = preprocessor.preprocess_data(df)
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(features, target)
    
    # Store in session state
    st.session_state['X_train'] = X_train
    st.session_state['y_train'] = y_train
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['feature_names'] = feature_names
    
    # Train models
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            predictor = load_predictor()
            
            # Update config to only train selected models
            predictor.config.models = {model: predictor.config.models[model] for model in selected_models}
            
            predictor.fit_models(X_train, y_train)
            
            # Evaluate models
            evaluator = EquipmentFailureEvaluator()
            results = []
            
            for model_name in selected_models:
                if model_name in predictor.models:
                    y_pred = predictor.predict(X_test, model_name)
                    metrics = evaluator.evaluate_model(y_test.values, y_pred, model_name)
                    results.append(metrics)
            
            # Store results
            st.session_state['predictor'] = predictor
            st.session_state['results'] = results
            st.session_state['evaluator'] = evaluator
            
            st.success(f"✅ Successfully trained {len(selected_models)} models!")
    
    # Show training results
    if 'results' in st.session_state:
        st.subheader("📊 Training Results")
        
        results_df = pd.DataFrame(st.session_state['results'])
        
        # Create comparison table
        st.dataframe(
            results_df[['model', 'mae', 'rmse', 'r2', 'mape']].round(3),
            use_container_width=True
        )
        
        # Best model
        best_model = results_df.loc[results_df['mae'].idxmin()]
        st.success(f"🏆 Best Model: {best_model['model']} (MAE: {best_model['mae']:.2f} hours)")


def show_predictions():
    """Show predictions tab."""
    st.header("📈 Real-time Predictions")
    
    if 'predictor' not in st.session_state:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    predictor = st.session_state['predictor']
    feature_names = st.session_state['feature_names']
    
    st.subheader("🔮 Equipment Failure Prediction")
    
    # Input form
    with st.form("prediction_form"):
        st.write("Enter sensor readings to predict time to failure:")
        
        col1, col2 = st.columns(2)
        
        sensor_values = {}
        for i, feature in enumerate(feature_names):
            if i % 2 == 0:
                with col1:
                    sensor_values[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        min_value=0.0,
                        max_value=1000.0,
                        value=75.0,
                        step=0.1
                    )
            else:
                with col2:
                    sensor_values[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        min_value=0.0,
                        max_value=1000.0,
                        value=35.0,
                        step=0.1
                    )
        
        submitted = st.form_submit_button("🔮 Predict Time to Failure", type="primary")
        
        if submitted:
            # Create input DataFrame
            input_df = pd.DataFrame([sensor_values])
            
            # Make predictions
            predictions = predictor.predict(input_df)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            for i, (model_name, pred) in enumerate(predictions.items()):
                ttf = pred[0]
                
                # Determine risk level
                if ttf < 50:
                    risk_level = "🔴 Critical"
                    risk_color = "danger-metric"
                elif ttf < 200:
                    risk_level = "🟡 High"
                    risk_color = "warning-metric"
                elif ttf < 500:
                    risk_level = "🟠 Medium"
                    risk_color = "warning-metric"
                else:
                    risk_level = "🟢 Low"
                    risk_color = "success-metric"
                
                with col1 if i == 0 else col2 if i == 1 else col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{model_name.replace('_', ' ').title()}</h4>
                        <p class="{risk_color}">{ttf:.1f} hours</p>
                        <p>{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show average prediction
            avg_pred = np.mean([pred[0] for pred in predictions.values()])
            st.metric("Average Prediction", f"{avg_pred:.1f} hours")
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            if avg_pred < 50:
                st.error("🚨 CRITICAL RISK: Immediate maintenance required!")
            elif avg_pred < 200:
                st.warning("⚠️ HIGH RISK: Schedule maintenance within 1 week")
            elif avg_pred < 500:
                st.info("ℹ️ MEDIUM RISK: Schedule maintenance within 1 month")
            else:
                st.success("✅ LOW RISK: Continue monitoring")


def show_business_impact():
    """Show business impact tab."""
    st.header("💼 Business Impact Analysis")
    
    if 'results' not in st.session_state:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    results = st.session_state['results']
    results_df = pd.DataFrame(results)
    
    # Business metrics
    st.subheader("💰 Cost Savings Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_model = results_df.loc[results_df['mae'].idxmin()]
        st.metric(
            "Maintenance Savings",
            f"${best_model['maintenance_cost_savings']:.0f}",
            help="Estimated annual maintenance cost savings"
        )
    
    with col2:
        st.metric(
            "Downtime Reduction",
            f"{best_model['downtime_reduction_hours']:.0f} hours",
            help="Estimated downtime reduction per year"
        )
    
    with col3:
        st.metric(
            "Total Cost Savings",
            f"${best_model['total_cost_savings']:.0f}",
            help="Total estimated annual cost savings"
        )
    
    with col4:
        st.metric(
            "Detection Rate",
            f"{best_model['detection_rate']:.1%}",
            help="Critical failure detection rate"
        )
    
    # Business impact charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost savings comparison
        fig = px.bar(
            results_df,
            x='model',
            y='total_cost_savings',
            title='Total Cost Savings by Model',
            labels={'total_cost_savings': 'Cost Savings ($)', 'model': 'Model'},
            color='total_cost_savings',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detection rate vs false alarm rate
        fig = px.scatter(
            results_df,
            x='false_alarm_rate',
            y='detection_rate',
            size='total_cost_savings',
            hover_name='model',
            title='Detection Rate vs False Alarm Rate',
            labels={'detection_rate': 'Detection Rate', 'false_alarm_rate': 'False Alarm Rate'},
            color='total_cost_savings',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ROI Analysis
    st.subheader("📊 Return on Investment (ROI)")
    
    # Simulate ROI calculation
    implementation_cost = 50000  # Estimated implementation cost
    annual_savings = best_model['total_cost_savings']
    roi_percentage = (annual_savings - implementation_cost) / implementation_cost * 100
    payback_period = implementation_cost / annual_savings if annual_savings > 0 else float('inf')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Implementation Cost", f"${implementation_cost:,}")
    
    with col2:
        st.metric("Annual Savings", f"${annual_savings:.0f}")
    
    with col3:
        st.metric("ROI", f"{roi_percentage:.1f}%")
    
    if payback_period < 2:
        st.success(f"✅ Payback period: {payback_period:.1f} years - Excellent investment!")
    elif payback_period < 5:
        st.info(f"ℹ️ Payback period: {payback_period:.1f} years - Good investment")
    else:
        st.warning(f"⚠️ Payback period: {payback_period:.1f} years - Consider carefully")


def show_model_analysis():
    """Show model analysis tab."""
    st.header("🔍 Model Analysis")
    
    if 'predictor' not in st.session_state or 'results' not in st.session_state:
        st.warning("Please train models first in the Model Training tab.")
        return
    
    predictor = st.session_state['predictor']
    results = st.session_state['results']
    results_df = pd.DataFrame(results)
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    # Performance metrics
    metrics_to_show = ['mae', 'rmse', 'r2', 'mape']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric.upper() for metric in metrics_to_show],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    for i, metric in enumerate(metrics_to_show):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Bar(
                x=results_df['model'],
                y=results_df[metric],
                name=metric.upper(),
                marker_color=colors[i],
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Model Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    # Get feature importance from best model
    best_model_name = results_df.loc[results_df['mae'].idxmin(), 'model']
    
    if best_model_name in predictor.models and hasattr(predictor.models[best_model_name], 'feature_importances_'):
        importance_df = predictor.get_feature_importance(best_model_name)
        
        fig = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title=f'Feature Importance - {best_model_name.replace("_", " ").title()}',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")
    
    # Model details
    st.subheader("📋 Model Details")
    
    for _, result in results_df.iterrows():
        with st.expander(f"{result['model'].replace('_', ' ').title()}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Performance Metrics:**")
                st.write(f"- MAE: {result['mae']:.2f} hours")
                st.write(f"- RMSE: {result['rmse']:.2f} hours")
                st.write(f"- R²: {result['r2']:.3f}")
                st.write(f"- MAPE: {result['mape']:.1f}%")
            
            with col2:
                st.write("**Business Metrics:**")
                st.write(f"- Cost Savings: ${result['total_cost_savings']:.0f}")
                st.write(f"- Detection Rate: {result['detection_rate']:.1%}")
                st.write(f"- False Alarm Rate: {result['false_alarm_rate']:.1%}")
                st.write(f"- Risk Accuracy: {result['risk_classification_accuracy']:.1%}")


if __name__ == "__main__":
    main()
