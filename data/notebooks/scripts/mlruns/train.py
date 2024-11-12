import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature

class IrisDataProcessor:
    def __init__(self):
        self.iris = load_iris()
        self.df = None
        self.scaler = StandardScaler()

    def prepare_data(self):
        self.df = pd.DataFrame(data=np.c_[self.iris['data'], self.iris['target']],
                               columns=self.iris['feature_names'] + ['target'])
        features = self.df.drop(columns=['target'])
        scaled_features = self.scaler.fit_transform(features)
        scaled_df = pd.DataFrame(scaled_features, columns=self.iris['feature_names'])
        self.df[self.iris['feature_names']] = scaled_df

        X = self.df[self.iris['feature_names']]
        y = self.df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_feature_stats(self):
        return self.df.describe()

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier()
        }
        mlflow.set_tracking_uri("http://localhost:5000")

    def run_experiment(self):
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data()
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')

                mlflow.log_param("model", model_name)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                input_example = X_train[:5]
                signature = infer_signature(X_train, predictions[:5])
                mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)

    def log_results(self):
        pass

def main():
    processor = IrisDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data()
    experiment = IrisExperiment(processor)
    experiment.run_experiment()

if __name__ == "__main__":
    main()
