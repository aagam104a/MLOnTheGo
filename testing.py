import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Adding custom CSS to improve UI
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stFileUploader {
        padding: 10px;
        margin: 20px 0;
    }
    h1, h2, h3, h4 {
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .report {
        background-color: #e8f4e8;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Step 1: Class for Model Selection and Management
class ModelManager:
    def __init__(self):
        # Dictionary of available models (expandable for future models)
        self.models = {
            "Random Forest": RandomForestClassifier()
        }
    
    def get_model(self, model_name):
        """Return the selected model from the available models."""
        return self.models.get(model_name, None)

# Step 2: Class for Data Upload and DataFrame Management
class DataManager:
    def __init__(self):
        self.data = None
    
    def upload_data(self):
        """Allow the user to upload a CSV file and convert it to a DataFrame."""
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], help="Upload a CSV file for classification tasks")
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
            st.write("Data Preview", self.data.head())
        return self.data

# Step 3: Class for Feature and Target Selection
class FeatureTargetSelector:
    def __init__(self, data):
        self.data = data
        self.features = None
        self.target = None
    
    def select_features_and_target(self):
        """Allow the user to select features and target from the DataFrame."""
        if self.data is not None:
            columns = self.data.columns.tolist()
            self.features = st.multiselect("Select Features", options=columns)
            self.target = st.selectbox("Select Target", options=columns)
        return self.features, self.target

    def show_selected(self):
        """Display the selected features and target."""
        if self.features and self.target:
            selection = {"Features": self.features, "Target": self.target}
            st.write("Selected Features and Target:")
            st.json(selection)
        else:
            st.write("Please select both features and target.")

# Step 4: Class for Model Training and Prediction
class ModelTrainer:
    def __init__(self, model, data, features, target):
        self.model = model
        self.data = data
        self.features = features
        self.target = target
        self.label_encoder = LabelEncoder()
        self.label_mapping = {}

    def label_encode_target(self):
        """Label encode the target if it's not numeric."""
        if not pd.api.types.is_numeric_dtype(self.data[self.target]):
            st.write("Target variable is not numeric. Applying label encoding.")
            self.data[self.target] = self.label_encoder.fit_transform(self.data[self.target])
            self.label_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
            st.write("Label Encoding Mapping:")
            st.json(self.label_mapping)

    def train_and_predict(self):
        """Train the model on the selected data and generate predictions."""
        if self.data is not None and self.features and self.target:
            self.label_encode_target()
            
            X = self.data[self.features]
            y = self.data[self.target]
            
            # Split data into training and testing sets (80-20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = self.model.predict(X_test)
            
            # Show accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Show classification report
            report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_ if self.label_mapping else None)
            st.write("Classification Report:")
            st.text(report)

            # Return predictions and actual values for further evaluation
            return y_pred, y_test
        else:
            st.write("Please upload data and select features/target before training.")

# Main Function to Run the App
def main():
    add_custom_css()  # Apply custom CSS for UI improvement
    st.title("MLOnTheGo - Train and Predict with Random Forest")
    
    # Step 1: Model Selection
    st.subheader("Step 1: Choose Model")
    model_manager = ModelManager()
    selected_model = model_manager.get_model("Random Forest")
    
    # Step 2: Upload CSV and Convert to DataFrame
    st.subheader("Step 2: Upload Data")
    data_manager = DataManager()
    df = data_manager.upload_data()

    # Step 3: Select Features and Target
    st.subheader("Step 3: Select Features and Target")
    if df is not None:
        selector = FeatureTargetSelector(df)
        features, target = selector.select_features_and_target()
        selector.show_selected()
    
    # Step 4: Train and Predict
    st.subheader("Step 4: Train the Model and Predict")
    if st.button("Train and Predict"):
        if df is not None and features and target:
            trainer = ModelTrainer(selected_model, df, features, target)
            trainer.train_and_predict()
        else:
            st.write("Please upload data and select both features and target.")

# Run the Streamlit app
if __name__ == "__main__":
    main()