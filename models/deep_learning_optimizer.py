import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
import numpy as np

class DeepLearningOptimizer:
    """
    Optimizer for Deep Learning model with architecture and hyperparameter tuning.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
    def create_model(self, params):
        """
        Create model with given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            model: Compiled Keras model
        """
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Dense(
            params['first_layer_units'],
            activation='relu',
            input_dim=self.input_dim
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(params['dropout_rate']))
        
        # Hidden layers
        for units in params['hidden_layers']:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(params['dropout_rate']))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mae'
        )
        
        return model
    
    def optimize_architecture(self, X_train, y_train, X_val, y_val):
        """
        Find optimal model architecture using grid search.
        
        Returns:
            dict: Best architecture parameters
        """
        architectures = [
            {
                'first_layer_units': 512,
                'hidden_layers': [256, 128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            },
            {
                'first_layer_units': 256,
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.4,
                'learning_rate': 0.001
            },
            {
                'first_layer_units': 128,
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.5,
                'learning_rate': 0.001
            }
        ]
        
        best_val_loss = float('inf')
        best_architecture = None
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        for arch in architectures:
            model = self.create_model(arch)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_architecture = arch
        
        return best_architecture
    
    def fine_tune_hyperparameters(self, best_architecture, X_train, y_train, X_val, y_val):
        """
        Fine-tune hyperparameters of the best architecture.
        
        Returns:
            dict: Optimized hyperparameters
        """
        learning_rates = [0.0001, 0.0005, 0.001, 0.005]
        dropout_rates = [0.2, 0.3, 0.4, 0.5]
        batch_sizes = [16, 32, 64]
        
        best_val_loss = float('inf')
        best_params = None
        
        for lr in learning_rates:
            for dr in dropout_rates:
                for bs in batch_sizes:
                    params = best_architecture.copy()
                    params['learning_rate'] = lr
                    params['dropout_rate'] = dr
                    
                    model = self.create_model(params)
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=bs,
                        callbacks=[
                            EarlyStopping(patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
                        ],
                        verbose=0
                    )
                    
                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            **params,
                            'batch_size': bs
                        }
        
        return best_params