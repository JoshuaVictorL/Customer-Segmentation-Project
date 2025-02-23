import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from customer_segmentation.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load transformed data
        data = pd.read_csv(self.config.transformed_data_path)

        # KMeans Clustering
        kmeans = KMeans(n_clusters=4, random_state=self.config.random_state, n_init=10)
        data['Cluster'] = kmeans.fit_predict(data.drop('User_ID', axis=1))

        os.makedirs(os.path.dirname(self.config.clustered_data_path), exist_ok=True)
        data[['User_ID', 'Cluster']].to_csv(self.config.clustered_data_path, index=False)

        # Prepare for Random Forest
        X = data.drop(['User_ID', 'Cluster'], axis=1)
        y = data['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        # Save the test dataset for evaluation
        test_data = X_test.copy()
        test_data['Cluster'] = y_test
        test_data.to_csv(self.config.test_data_path, index=False)

        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Save the trained model
        joblib.dump(rf, self.config.model_save_path)