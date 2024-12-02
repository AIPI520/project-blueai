from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb

class XGBoostOptimizer:
    """
    Optimizer for XGBoost model with hyperparameter tuning.
    """
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def optimize_hyperparameters(self, cv=5):

        """
        Perform hyperparameter optimization using RandomizedSearchCV.
        
        Returns:
            dict: Best parameters
            
        """

        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'reg_alpha': [0, 0.1, 1, 2, 5],
            'reg_lambda': [0, 0.1, 1, 2, 5]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=100,
            scoring='neg_mean_absolute_error',
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        return random_search.best_params_
    
    def fine_tune_model(self, best_params):
        """
        Fine-tune the model around the best parameters.
        
        Args:
            best_params: Initial best parameters
            
        Returns:
            dict: Fine-tuned parameters
        """
        fine_tune_params = {
            'n_estimators': [
                best_params['n_estimators'] - 50,
                best_params['n_estimators'],
                best_params['n_estimators'] + 50
            ],
            'max_depth': [
                best_params['max_depth'] - 1,
                best_params['max_depth'],
                best_params['max_depth'] + 1
            ],
            'learning_rate': [
                best_params['learning_rate'] * 0.8,
                best_params['learning_rate'],
                best_params['learning_rate'] * 1.2
            ]
        }
        
        grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(**best_params),
            param_grid=fine_tune_params,
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_