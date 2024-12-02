import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    """
    Advanced feature engineering pipeline with feature selection and generation.
    Specifically designed for housing price prediction with given features.
    """
    def __init__(self):
        self.poly_features = None
        self.feature_selector = None
        self.selected_features = None
        
    def create_domain_features(self, X):
        """
        Create domain-specific features for housing data.
        
        Args:
            X (DataFrame): Input features with housing-specific columns
            
        Returns:
            DataFrame: Enhanced features
        """
        X = X.copy()
        
        # Price per square foot (if SaleAmt is present)
        if 'SaleAmt' in X.columns and 'LivingSqFt' in X.columns:
            X['Price_per_SqFt'] = X['SaleAmt'] / (X['LivingSqFt'] + 1e-5)
        
        # Living space ratios
        X['Rooms_per_SqFt'] = X['RoomsNbr'] / (X['LivingSqFt'] + 1e-5)
        X['Bedrooms_per_Room'] = X['BedroomsNbr'] / (X['RoomsNbr'] + 1e-5)
        X['Bath_per_Bedroom'] = X['TotalBath'] / (X['BedroomsNbr'] + 1e-5)
        
        # Property age interactions
        X['Age_squared'] = X['Age'] ** 2
        X['Age_LivingSqFt'] = X['Age'] * X['LivingSqFt']
        
        # Location-based features
        X['Park_Access'] = (X['Park_Pct_5'] + X['Park_Pct_10'] + X['Park_Pct_15']) / 3
        X['Park_Proximity_Weight'] = (X['Park_Pct_5'] * 0.5 + 
                                    X['Park_Pct_10'] * 0.3 + 
                                    X['Park_Pct_15'] * 0.2)
        
        # Economic indicators
        X['Income_Distance_Ratio'] = X['Med.Income'] / (X['Dist_cbd'] + 1e-5)
        
        # Education and demographics interaction
        X['Educ_Income_Inter'] = X['Educ_High'] * X['Med.Income']
        X['Youth_Education_Inter'] = X['Perc_18'] * X['Educ_High']
        
        # Property characteristics
        X['Total_Rooms'] = X['RoomsNbr'] + X['BedroomsNbr']
        X['Room_Bath_Ratio'] = X['Total_Rooms'] / (X['TotalBath'] + 1e-5)
        
        # Area utilization
        X['Living_Area_Ratio'] = X['LivingSqFt'] / (X['LtArea'] + 1e-5)
        
        # Amenity scores
        X['Amenity_Score'] = (X['Garage'] + X['FirePl'] + 
                             X['TotalBath']/2 + X['BedroomsNbr']/3)
        
        return X
    
    def create_polynomial_features(self, X, degree=2):
        """
        Create polynomial features for numerical columns.
        
        Args:
            X: Input features
            degree: Polynomial degree
            
        Returns:
            array: Enhanced features with polynomial interactions
        """
        # Select only numeric columns for polynomial features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        try:
            if self.poly_features is None:
                self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = self.poly_features.fit_transform(X_numeric)
                feature_names = self.poly_features.get_feature_names_out(numeric_cols)
                return pd.DataFrame(poly_features, columns=feature_names)
            
            poly_features = self.poly_features.transform(X_numeric)
            feature_names = self.poly_features.get_feature_names_out(numeric_cols)
            return pd.DataFrame(poly_features, columns=feature_names)
            
        except Exception as e:
            print(f"Error in polynomial feature creation: {str(e)}")
            return X
    
    def select_features(self, X, y, k=20):
        """
        Select top k most important features.
        
        Args:
            X: Input features
            y: Target variable
            k: Number of features to select
            
        Returns:
            array: Selected features
        """
        try:
            # Ensure k is not larger than the number of features
            k = min(k, X.shape[1])
            
            if self.feature_selector is None:
                self.feature_selector = SelectKBest(score_func=f_regression, k=k)
                return self.feature_selector.fit_transform(X, y)
            return self.feature_selector.transform(X)
        except Exception as e:
            print(f"Error in feature selection: {str(e)}")
            return X