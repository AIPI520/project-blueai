# main.py
from data_processing.preprocessor import DataPreprocessor
from models.xgboost_model import XGBoostModel
from models.deep_learning_model import DeepLearningModel
from models.xgboost_optimizer import XGBoostOptimizer
from models.deep_learning_optimizer import DeepLearningOptimizer
from feature_engineering.feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    """Main execution function for the housing price prediction pipeline."""
    
    # Initialize preprocessor and load data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data('Housing01.csv')
    
    # Print columns for verification
    print("\nAvailable columns:", data.columns.tolist())
    
    # Preprocess data
    X_scaled, y = preprocessor.preprocess_features(data)
    print("\nFeatures after preprocessing:", X_scaled.columns.tolist())
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Feature Engineering and Optimization
    print("\nPerforming feature engineering...")
    feature_engineer = FeatureEngineer()
    
    # Create enhanced features
    X_train_enhanced = feature_engineer.create_domain_features(X_train)
    X_val_enhanced = feature_engineer.create_domain_features(X_val)
    X_test_enhanced = feature_engineer.create_domain_features(X_test)
    
    # Create polynomial features
    print("Generating polynomial features...")
    X_train_poly = feature_engineer.create_polynomial_features(X_train_enhanced)
    X_val_poly = feature_engineer.create_polynomial_features(X_val_enhanced)
    X_test_poly = feature_engineer.create_polynomial_features(X_test_enhanced)
    
    # Feature selection
    print("Performing feature selection...")
    X_train_final = feature_engineer.select_features(X_train_poly, y_train)
    X_val_final = feature_engineer.select_features(X_val_poly, y_val)
    X_test_final = feature_engineer.select_features(X_test_poly, y_test)
    
    # XGBoost Optimization
    print("\nOptimizing XGBoost model...")
    xgb_optimizer = XGBoostOptimizer(X_train_final, y_train, X_val_final, y_val)
    xgb_best_params = xgb_optimizer.optimize_hyperparameters()
    print("Best XGBoost parameters:", xgb_best_params)
    
    # Fine-tune XGBoost
    print("Fine-tuning XGBoost model...")
    xgb_final_params = xgb_optimizer.fine_tune_model(xgb_best_params)
    print("Final XGBoost parameters:", xgb_final_params)
    
    # Deep Learning Optimization
    print("\nOptimizing Deep Learning model...")
    dl_optimizer = DeepLearningOptimizer(X_train_final.shape[1])
    dl_best_architecture = dl_optimizer.optimize_architecture(
        X_train_final, y_train, X_val_final, y_val
    )
    print("Best DL architecture:", dl_best_architecture)
    
    # Fine-tune Deep Learning
    print("Fine-tuning Deep Learning model...")
    dl_final_params = dl_optimizer.fine_tune_hyperparameters(
        dl_best_architecture, X_train_final, y_train, X_val_final, y_val
    )
    print("Final DL parameters:", dl_final_params)
    
    # Train final XGBoost model with optimized parameters
    print("\nTraining final XGBoost model...")
    xgb_model = XGBoostModel()
    xgb_model.model.set_params(**xgb_final_params)
    xgb_model.train(X_train_final, y_train)
    
    # Train final Deep Learning model with optimized parameters
    print("\nTraining final Deep Learning model...")
    dl_model = DeepLearningModel(input_dim=X_train_final.shape[1])
    dl_model.train(
        X_train=X_train_final,
        y_train=y_train,
        params=dl_final_params,
        X_val=X_val_final,
        y_val=y_val,
        epochs=100
    )
    
    # Save models
    xgb_model.save_model()
    dl_model.save_model()
    
    xgb_model = XGBoostModel()
    xgb_model.load_model()

    dl_model = DeepLearningModel(input_dim=20)
    dl_model.load_model()

    # Evaluate both models
    print("\nEvaluating models on test set...")
    xgb_metrics = xgb_model.evaluate(X_test_final, y_test)
    dl_metrics = dl_model.evaluate(X_test_final, y_test)
    
    print("\nModel Evaluation Results:")
    print(f"XGBoost MAE: {xgb_metrics['mae']}")
    print(f"XGBoost R2 Score: {xgb_metrics['r2']}")
    print(f"Deep Learning MAE: {dl_metrics['mae']}")
    print(f"Deep Learning R2 Score: {dl_metrics['r2']}")

if __name__ == "__main__":
    main()