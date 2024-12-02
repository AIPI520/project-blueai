import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Data preprocessing pipeline for the housing price prediction model."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
    
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            DataFrame: Loaded data
        """
        return pd.read_csv(file_path)
    
    def preprocess_features(self, data):
        """
        Preprocess the feature data.
        
        Args:
            data (DataFrame): Raw data
            
        Returns:
            tuple: Processed features and target variable
        """
        # Separate features and target
        X = data.drop(['Price.Annual', 'tract.bg'], axis=1)
        y = data['Price.Annual']
        
        # Scale features while preserving column names
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled, y