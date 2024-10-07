import streamlit as st

# Class to handle the selection of models
class ModelSelector:
    def __init__(self, models):
        """
        Initializes the ModelSelector with a list of models.
        
        Args:
            models (list): List of model names to display in the dropdown.
        """
        self.models = models
        self.selected_model = None

    def render_dropdown(self):
        """
        Renders the model selection dropdown on the Streamlit interface.
        """
        self.selected_model = st.selectbox(
            label="Choose a Pretrained Model", 
            options=self.models
        )
        st.write(f"Selected Model: {self.selected_model}")

# Main app function
def main():
    st.title("MLOnTheGo")

    # Available model names (for testing purposes)
    model_names = ["Random Forest", "Logistic Regression", "XGBoost", "SVM", "KNN"]

    # Step 1: Model Selection
    model_selector = ModelSelector(model_names)
    model_selector.render_dropdown()

if __name__ == "__main__":
    main()