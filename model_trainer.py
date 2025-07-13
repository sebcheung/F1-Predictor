import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from joblib import dump
import os

class F1RacePointsTrainer:
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None

        os.makedirs(self.model_dir, exist_ok=True)

    # Load in training data and validation data
    def load_data(self):
        print("Loading training and validation data...")
        self.X_train = pd.read_csv(os.path.join(self.data_dir, "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(self.data_dir, "y_train.csv"))["race_points"]

        self.X_val = pd.read_csv(os.path.join(self.data_dir, "X_val.csv"))
        self.y_val = pd.read_csv(os.path.join(self.data_dir, "y_val.csv"))["race_points"]

        print(f"Loaded: X_train {self.X_train.shape}, X_val {self.X_val.shape}")

    # Build the model using XGBRegressor (aka regression line)
    def build_model(self):
        print("Initializing XGBoost regressor...")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            objective="reg:squarederror",
            random_state=42
        )

    # Train the model using X_train and y_train
    def train(self):
        print("Training model on race_points...")
        self.model.fit(self.X_train, self.y_train)
        print("Training complete.")

    # Evaluate and analyze the model's accuracy based on validation set from training
    def evaluate(self):
        print("Evaluating model on validation set...")
        preds = self.model.predict(self.X_val)

        rmse = root_mean_squared_error(self.y_val, preds)
        mae = mean_absolute_error(self.y_val, preds)

        print(f"Validation RMSE: {rmse:.3f}")
        print(f"Validation MAE:  {mae:.3f}")

    # Saving model into a joblib file in its own directory
    def save_model(self):
        model_path = os.path.join(self.model_dir, "xgb_race_points_model.joblib")
        dump(self.model, model_path)
        print(f"Model saved to: {model_path}")

    # Runs all the functions in an organized format
    def run(self):
        self.load_data()
        self.build_model()
        self.train()
        self.evaluate()
        self.save_model()


if __name__ == "__main__":
    trainer = F1RacePointsTrainer()
    trainer.run()