[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/A-vEqCXL)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=17390693)
# Housing Price Prediction Project

## Team Members
Zihao Yang, Tony Wang, Zejun Bai, Reina Shi

## Video Presentation
        
[Link to video presentation]

## Project Overview 
        
This project is designed to a pratical problem: housing price prediction. It implements and compares two different machine learning approaches for predicting housing prices: an XGBoost model (non-deep learning) and a neural network (deep learning) model. Both models are trained from scratch and optimized through feature engineering and hyperparameter tuning. 

## Dataset Description

This Dataset is housing price and characteristics data compiled by the City and County of Denver's Assessment Division through the Denver Open Data Catalog referred by a paper: *Valuing Curb Appeal* by [ Johnson, E.B., Tidwell, A. & Villupuram ]

**Citation:**
        
Johnson, E.B., Tidwell, A. & Villupuram, S.V. Valuing Curb Appeal. J Real Estate Finan Econ 60, 111–133 (2020). https://doi.org/10.1007/s11146-019-09713-z

**Link to the paper:**
[Valuing Curb Appeal](https://doi.org/10.1007/s11146-019-09713-z)

The dataset used in this project contains housing transaction data with various features including:        
- Physical characteristics (bedrooms, bathrooms, living space, etc.)
- Location attributes (distance to CBD, park proximity)
- Demographic information (median income, education levels)
- Historical data (year sold, age of property)
            
  The target variable is the annual price of the property. The dataset excludes the typical Kaggle housing datasets and provides a unique perspective on housing price prediction by incorporating both physical and socio-economic factors.

## Data Pipeline

### Data Preprocessing
1. Initial data loading and cleaning
2. Feature scaling using StandardScaler
3. Train-validation-test split (60-20-20)
4. Handling of missing values and outliers

### Feature Engineering
1. Construct Domain-specific features:
   - Price per square foot
   - Room and space utilization ratios
   - Age-related interactions
   - Location-based composite scores
   - Economic indicator ratios
   - Education and demographics interations
   - Property characteristics 
   - Area utilization
   - Amenity scores

2. Advanced feature generation:
   - Polynomial features (degree 2)   
   - Feature selection using SelectKBest. We selected most 20 important features and used f_regression as the scoring function, which is suitable for regression tasks. It calculates the correlation between each feature and the target y using an F-statistic.   

## Models

### XGBoost Model
- Non-deep learning approach
- Hyperparameter optimization using RandomizedSearchCV.      
           
  RandomizedSearchCV is a tool in scikit-learn that is used to perform hyperparameter optimization for machine learning models. Unlike GridSearchCV, which exhaustively tests all possible combinations of hyperparameters, RandomizedSearchCV randomly samples a predefined number of combinations from the hyperparameter space. This makes it more efficient and faster, especially when dealing with large hyperparameter spaces. In this part we consider optimize hyperparameters including:    
         
  1. n_estimators: Number of boosting round (trees).     
  2. max_depth: Maximum depth of trees.  
  3. learning_rate: Step size shrinkage.  
  4. min_child_weight: Minimum sum of weights of child nodes.    
  5. subsample: Fraction of samples used for training each tree.    
  6. colsample_bytree: Fraction of features used per tree.  
  7. gamma: Mimimum loss reduction to make a further split.   
  8. reg_alpha: L1 regularization term.   
  9. reg_lambda: L2 regularization term.  
- Fine-tuning process with GridSearchCV.    
            
  GridSearchCV is a tool in scikit-learn that performs an exhaustive search over a specified hyperparameter space for a machine learning model. It evaluates all possible combinations of hyperparameters specified by the user and determines the best combination based on a predefined performance metric. Key parameters optimized:     
  - n_estimators   
  - max_depth     
  - learning_rate    
- Final XGBoost Parameters with optimization and fine-tuning:     
        
  subsample: 1.0     
  reg_lambda: 0     
  reg_alpha: 2      
  n_estimators: 450     
  min_child_weight: 1      
  max_depth: 9    
  learning_rate: 0.2       
  gamma: 0     
  colsample_bytree: 0.9     

### Deep Learning Model
- Custom neural network built with TensorFlow/Keras.    
       
  We built a multiple layer neural network. The first layer is a fully connected layer with ReLU activation and batch normalization with a specific dropout rate. The hidden layere are a fully connected layers with ReLU activation and batch normalization with a specific dropout rate. We chose mae as the loss function and used Adam optimizer to optimize the parameters. It combines Benefits of SGD and RMSProp with smooth and accelerate training and helping adjust the learning rate for each parameter individually.    
- Trained from scratch (no transfer learning)
- Architecture optimization:
  - Multiple layer configurations
  - Dropout for regularization
  - Batch normalization
- Hyperparameter tuning:   
    
  We tried different combination of hyperparameters tuning of this deep learning model with different number of first layer units, different number of hidden layers with different units in each layer, different dropout rates and different learning rates. For the efficiency and preventing overfitting in finding the best hyperparameters, we used earlystopping method to stop training if validation loss does not improve for 10 epochs and ReduceLROnPlateau to reduce the learning rate by a factor of 0.2 if validation loss does not improve for 5 epochs.    
- Hyperparameter optimization:
    
  We set different learning rates, different dropout rates and different batch sizes and tried all combinations for these hyperparameters. Same as above, we tried early stopping method and ReduceLROnPlateau method for preventing overfitting and efficiency. 
- Final DL structure and parameters:   
         
  A 6 layer neural network with 1 first layer, 4 hidden layers and 1 output layer. There are 512 units in the first layer, 256, 128, 64, 32 units for the hidden layers with a 0.2 drop rate, 0,005 learning rate and 64 batch size. 
      
## Evaluation Strategy
- Primary metric: Mean Absolute Error (MAE).   

  MAE measures the average magnitude of errors between predicted and actual values, providing a straightforward interpretation of how far off the predictions are from the true values.    
- Secondary metric: R² Score. It measures how well the predictions of a model match the actual data by comparing it to a baseline model.    
- Separate validation set for hyperparameter tuning. 
- Independent test set for final evaluation.

## Model Performance on Test Set

### Before Hyperparameter Optimization
- XGBoost:
  - MAE: 4307.01
  - R² Score: 0.8424
- Deep Learning:
  - MAE: 4905.77
  - R² Score: 0.8121

### After Hyperparameter Optimization
- XGBoost:
  - MAE: 54.879
  - R² Score: 0.999
- Deep Learning:
  - MAE: 864.25
  - R² Score: 0.992

### Analysis

#### Expected
   
- XGBoost and Deep Learning method have a similar R² Score but different MAE score.  

  R² Score measures the proportion of variance in the target variable that the model explains and MAE measures the average absolute error between predicted and actual values. It is because though neural networks may struggle to optimize well for individual predictions resulting in larger errors for specific data points increasing MAE, it can still align well with the data's overall trend leading to a higher R² Score. 

- The MAE and R² Score have a significant improvement after the hyperparameter optimization in both XGBoost and Deep Learning.  
     
  It shows that finding the best parameters with sufficient techniques to prevent overfitting to get best performance in validation set will not only better performance of the model but not harm the generalization ability in further data in this problem.  

#### Unexpected
- XGBoost outperformed the deep learning model before and after the hyperparameter Optimization in both MAE and R² Score.     

  Possible Explanation: For this task, traditional approach with XGBoost performs better than neural network. This is potentially because this prediction involves tabular data with non-linear but relatively simple relationship between features. XGBoost is a gradient-boosted decision tree algorithm specifically designed for tabular data and it could handle feature interactions well. However, neural networks face challenge when applied to tabular data because of lack of feature interaction modeling though they can handle unstructed data well (audio, images).

#### Possible improvements:
- Use attention mechanism to capture the relationship between different features such as transformers. 
- Ensemble approaches for better generalization ability. 
- Extended hyperparameter optimization improving the model performance. 
- Additional data collection. 

## Project Structure
```
project/
│
├── data_processing/
│   └── preprocessor.py
├── feature_engineering/
│   └── feature_optimizer.py
├── models/
│   ├── xgboost_model.py
│   ├── deep_learning_model.py
│   ├── xgboost_optimizer.py
│   └── deep_learning_optimizer.py
├── main.py
├── inference.py
├── requirements.txt
├── Housing01.csv
└── README.md
```

## Running the Project

### Requirements
```bash
pip install -r requirements.txt
```

### Training Models
```bash
python main.py
```
This will:
1. Load and preprocess the data
2. Perform feature engineering
3. Train and optimize both models
4. Save the trained models
5. Output evaluation metrics

### Inference
```bash
streamlit python inference.py 
```
This will give a visual interface for inference. It can:
1. Load the saved models.
2. Process new data.
3. Generate predictions with both the XGBoost model and neural network model.

## License

MIT License

Copyright (c) [2024] [Zihao Yang, Tony Wang, Zejun Bai, Reina Shi]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.