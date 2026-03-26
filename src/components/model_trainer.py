import os
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.exception import Error
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Extra Trees": ExtraTreesClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "AdaBoost": AdaBoostClassifier(random_state=42),
            }

            params = {
                "Extra Trees": {
                    "n_estimators": [200, 400],
                    "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 3],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "n_estimators": [200, 400],
                    "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 3],
                    "max_features": ["sqrt", "log2"],
                },
                "Logistic Regression": {
                    "C": [0.1, 1.0, 3.0, 5.0],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                },
            }
        
            # Using utils.evaluate_models
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )
        
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.0:
                raise Exception("No best model found (Accuracy < 0.0)")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name} with accuracy {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f"Best Model & Accuracy: {best_model_name} - {best_model_score:.4f}")

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            
            return accuracy

        except Exception as e:
            raise Error(e)
