from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
import joblib

class XGBoostModel:
    """
    XGBoost regression model with cross-validation and model persistence capabilities.
    
    Attributes:
        model: Trained XGBoost model instance
        model_path (str): Path to save/load the model
    """
    
    def __init__(self, random_state=42, model_path="models/xg_model.pkl"):
        """
        Initialize XGBoost model with given parameters.
        
        Args:
            random_state (int): Random seed for reproducibility
            model_path (str): Path to save/load the model
        """
        self.model = XGBRegressor(random_state=random_state)
        self.model_path = model_path
    
    def train(self, X_train, y_train):
        """
        Train the XGBoost model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training target values
        """
        self.model.fit(X_train, y_train)
    
    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform cross-validation on the training data.
        
        Args:
            X_train: Training features
            y_train: Training target values
            cv (int): Number of cross-validation folds
            
        Returns:
            tuple: Mean MAE score and individual fold scores
        """
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        mae_scores = -cv_scores
        return mae_scores.mean(), mae_scores
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    def save_model(self):
        """Save the trained model to disk."""
        self.model.save_model(self.model_path)  # Save the model in XGBoost format
        
    def load_model(self):
        """Load a trained model from disk."""
        self.model = XGBRegressor()  # Re-initialize the model
        self.model.load_model(self.model_path)  # Load the model