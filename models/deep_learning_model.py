import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error,r2_score
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.metrics import MeanAbsoluteError

@register_keras_serializable()
def mae(y_true, y_pred):
    return MeanAbsoluteError()(y_true, y_pred)


class DeepLearningModel:
    """
    Deep Learning regression model using TensorFlow/Keras.
    
    Attributes:
        model: Trained Keras model instance
        model_path (str): Path to save/load the model
    """
    
    def __init__(self, input_dim, model_path="models/dl-model.keras"):
        """
        Initialize the deep learning model.
        
        Args:
            input_dim (int): Number of input features
            model_path (str): Path to save/load the model
        """
        self.model_path = model_path
        self.model = None
    
    def build_model(self, params):
        """
        Build the neural network with given parameters.
        
        Args:
            params (dict): Model parameters including architecture and hyperparameters
        """
        model = Sequential()
        
        # First layer
        model.add(Dense(params['first_layer_units'], 
                       activation='relu', 
                       input_shape=(self.input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
        # Hidden layers
        for units in params['hidden_layers']:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mae',
            metrics=['mae']
        )
        
        self.model = model
    
    def train(self, X_train, y_train, params, X_val=None, y_val=None, epochs=100):
        """
        Train the deep learning model.
        
        Args:
            X_train: Training features
            y_train: Training target values
            params: Model parameters including architecture and hyperparameters
            X_val: Validation features (optional)
            y_val: Validation target values (optional)
            epochs: Number of training epochs
        
        Returns:
            history: Training history
        """
        self.input_dim = X_train.shape[1]
        self.build_model(params)
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=params.get('batch_size', 32),
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
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    def save_model(self):
        """Save the trained model to disk."""
        self.model.save(self.model_path)
        
    def load_model(self):
        """Load a trained model from disk."""
        self.model = load_model(self.model_path)