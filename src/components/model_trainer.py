import os
import sys
from dataclasses import dataclass
from src.utils import evaluate_model

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )        
            models = {
            'Linear Regression': LinearRegression(),
            'K-Neighbors Regressor': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'Random Forest Regressor': RandomForestRegressor(),
            'XGBRegressor': XGBRegressor(),
            'CatBoosting Regressor': CatBoostRegressor(verbose=False),
            'AdaBoost Regressor': AdaBoostRegressor()
            }
            params = {
            'Linear Regression': {
            # No major hyperparameters for basic LinearRegression
            },
            'K-Neighbors Regressor': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },

            'Decision Tree': {
                'criterion': ['squared_error', 'friedman_mse'],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },

            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'max_depth': [3, 5, 7]
            },

            'Random Forest Regressor': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]
            },

            'XGBRegressor': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },

            'CatBoosting Regressor': {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6, 8]
            },

            'AdaBoost Regressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'loss': ['linear', 'square', 'exponential']
            }
        }

            model_report:dict = evaluate_model(X_train=X_train, 
                                               y_train=y_train, 
                                               X_test=X_test,
                                               y_test=y_test,
                                               models=models,
                                               param=params)
            
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < .6:
                raise CustomException('No best model found')
            
            logging.info(f'Best found model on both training and testing dataset')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
