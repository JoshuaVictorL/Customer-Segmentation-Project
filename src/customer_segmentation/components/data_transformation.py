import pandas as pd
from sklearn.preprocessing import StandardScaler
from customer_segmentation.entity.config_entity import DataTransformationConfig
from customer_segmentation import logger

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        """Load data from the provided CSV path."""
        self.df = pd.read_csv(self.config.data_path)

    def aggregate_purchase_behavior(self):
        """Aggregate purchase behavior per user."""
        self.df_cluster = self.df.groupby('User_ID').agg(
            total_spent=('Purchase', 'sum'),
            avg_spent=('Purchase', 'mean'),
            spending_variability=('Purchase', 'std')
        ).reset_index()

    def merge_demographic_data(self):
        """Merge user demographic data."""
        df_demographics = self.df[['User_ID', 'Age', 'Occupation', 'City_Category', 
                                   'Stay_In_Current_City_Years', 'Marital_Status']].drop_duplicates('User_ID')
        self.df_cluster = self.df_cluster.merge(df_demographics, on='User_ID', how='left')

    def encode_categorical_variables(self):
        """Map and encode categorical variables."""
        age_mapping = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
        stay_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}

        self.df_cluster['Age'] = self.df_cluster['Age'].map(age_mapping)
        self.df_cluster['Stay_In_Current_City_Years'] = self.df_cluster['Stay_In_Current_City_Years'].map(stay_mapping)

        # One-hot encode City_Category
        self.df_cluster = pd.get_dummies(self.df_cluster, columns=['City_Category'], prefix='City')

    def process_occupation_data(self):
        """Normalize occupation frequency and process occupation column."""
        occupation_counts = self.df_cluster['Occupation'].value_counts(normalize=True)
        self.df_cluster['Occupation_Freq'] = self.df_cluster['Occupation'].map(occupation_counts)
        self.df_cluster.drop(columns=['Occupation'], inplace=True)

    def product_purchase_behavior(self):
        """Analyze product purchase patterns."""
        df_product = self.df.groupby('User_ID').agg(
            top_category=('Product_Category', lambda x: x.value_counts().idxmax()),  # Most purchased category
            unique_categories=('Product_Category', 'nunique')  # Number of unique categories purchased
        ).reset_index()

        # Merge product behavior data into the main DataFrame
        self.df_cluster = pd.merge(self.df_cluster, df_product, on='User_ID', how='left')

        # Group rare categories into 'Other'
        rare_categories = self.df_cluster['top_category'].value_counts()[self.df_cluster['top_category'].value_counts() < 1000].index
        self.df_cluster['top_category'] = self.df_cluster['top_category'].apply(lambda x: 'Other' if x in rare_categories else x)

        # One-hot encode 'top_category'
        self.df_cluster = pd.get_dummies(self.df_cluster, columns=['top_category'], prefix='Category')

    def scale_features(self):
        """Scale numeric features using StandardScaler."""
        # Exclude 'User_ID' from scaling
        features_to_scale = self.df_cluster.drop(['User_ID'], axis=1)

        # Initialize the scaler
        scaler = StandardScaler()

        # Apply scaling
        scaled_features = scaler.fit_transform(features_to_scale)

        # Convert back to DataFrame
        self.df_cluster_scaled = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
        
        # Add back 'User_ID'
        self.df_cluster_scaled['User_ID'] = self.df_cluster['User_ID'].values

    def save_transformed_data(self):
        """Save the transformed and scaled dataset to the specified directory."""
        transformed_path = f"{self.config.root_dir}/transformed_scaled_data.csv"
        self.df_cluster_scaled.to_csv(transformed_path, index=False)

    def run_transformation(self):
        """Run all transformation steps in order."""
        self.load_data()
        self.aggregate_purchase_behavior()
        self.merge_demographic_data()
        self.encode_categorical_variables()
        self.process_occupation_data()
        self.product_purchase_behavior()
        self.scale_features()
        self.save_transformed_data()