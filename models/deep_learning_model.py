import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

class DeepLearningModel:
    """
    Deep Learning regression model using TensorFlow/Keras.
    
    Attributes:
        model: Trained Keras model instance
        model_path (str): Path to save/load the model
    """
    
    def __init__(self, input_dim, model_path="models/regression_nn_model.h5"):
        """
        Initialize the deep learning model.
        
        Args:
            input_dim (int): Number of input features
            model_path (str): Path to save/load the model
        """
        self.model_path = model_path
        self.model = self._build_model(input_dim)
    
    def _build_model(self, input_dim):
        """
        Build the neural network architecture.
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            model: Compiled Keras model
        """
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mae', 
                     metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the deep learning model.
        
        Args:
            X_train: Training features
            y_train: Training target values
            validation_split (float): Fraction of training data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            history: Training history
        """
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
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
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    def save_model(self):
        """Save the trained model to disk."""
        self.model.save(self.model_path)
        
    def load_model(self):
        """Load a trained model from disk."""
        self.model = load_model(self.model_path)