from pathlib import Path
from typing import Any

import dill
import pandas as pd

from src.exception import Error
from src.logger import logging


class PredictPipeline:
    RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

    def __init__(
        self,
        model_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
    ):
        project_root = Path(__file__).resolve().parents[2]

        self.model_path = Path(model_path) if model_path is not None else project_root / "artifacts" / "model.pkl"
        self.preprocessor_path = (
            Path(preprocessor_path)
            if preprocessor_path is not None
            else project_root / "artifacts" / "preprocessor.pkl"
        )

        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            with open(self.model_path, "rb") as file_obj:
                return dill.load(file_obj)

        except Exception as e:
            raise Error(e)

    def load_preprocessor(self):
        try:
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")

            with open(self.preprocessor_path, "rb") as file_obj:
                preprocessor = dill.load(file_obj)

            # If the saved object is the wrapper class instead of the fitted transformer,
            # rebuild the actual transformer here.
            if not hasattr(preprocessor, "transform"):
                from src.components.data_transformation import DataTransformation

                if isinstance(preprocessor, DataTransformation):
                    logging.warning(
                        "Loaded preprocessor is a DataTransformation object. "
                        "Rebuilding the actual transformer."
                    )
                    preprocessor = preprocessor.get_data_transformer_object()
                else:
                    raise TypeError(
                        f"Loaded preprocessor from {self.preprocessor_path} does not support transform()."
                    )

            return preprocessor

        except Exception as e:
            raise Error(e)

    def _prepare_input(self, input_data: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(input_data, dict) or not input_data:
            raise ValueError("input_data must be a non-empty dictionary")

        df = pd.DataFrame([input_data])

        expected_columns = [
            "Age",
            "Symptoms",
            "Symptom_Count",
            "Doctor_Notes",
            "Gender",
            "Lifestyle",
            "Medical_History",
            "Medications",
            "Lab_Reports",
            "Diagnosis",
        ]

        for column in expected_columns:
            if column not in df.columns:
                if column in {"Age", "Symptom_Count"}:
                    df[column] = 0
                else:
                    df[column] = ""

        symptoms = df.loc[0, "Symptoms"]
        if pd.notnull(symptoms) and str(symptoms).strip():
            df.loc[0, "Symptom_Count"] = len([s for s in str(symptoms).split(",") if s.strip()])
        elif "Symptom_Count" in input_data and pd.notnull(input_data["Symptom_Count"]):
            df.loc[0, "Symptom_Count"] = input_data["Symptom_Count"]
        else:
            df.loc[0, "Symptom_Count"] = 0

        return df[expected_columns]

    def predict(self, input_data: dict[str, Any]):
        try:
            input_df = self._prepare_input(input_data)
            transformed_input = self.preprocessor.transform(input_df)
            prediction = self.model.predict(transformed_input)[0]

            try:
                return self.RISK_LABELS[int(prediction)]
            except Exception:
                return prediction

        except Exception as e:
            logging.exception("Error during prediction")
            raise Error(e)


if __name__ == "__main__":
    pipeline = PredictPipeline()
    print(
        pipeline.predict(
            {
                "Age": 40,
                "Symptoms": "fever,cough",
                "Doctor_Notes": "Routine checkup",
                "Gender": "Male",
                "Lifestyle": "High",
                "Medical_History": "No",
                "Medications": "No",
                "Lab_Reports": "No",
                "Diagnosis": "No",
            }
        )
    )