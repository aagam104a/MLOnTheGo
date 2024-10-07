# MLOnTheGo

**MLOnTheGo** is a lightweight machine learning interface that allows users to quickly train and evaluate machine learning models using pre-trained models on their own datasets. Built with Streamlit, the application offers a simple user interface where users can upload datasets, choose features, set train-test splits, and generate predictions that can be downloaded as CSV files.

## Features

1. **Model Selection**: Users can select from various pre-trained machine learning models through a dropdown menu.
2. **CSV Upload**: Users can upload their dataset in CSV format, which will be automatically converted into a Pandas DataFrame.
3. **Feature and Target Selection**: After uploading the data, users can select their desired features and target variable(s) using a multi-select box.
4. **Train-Test Split**: Users can define the split ratio for training and testing data (e.g., 80-20, 70-30) based on their input.
5. **Prediction Output**: Once the model is selected and the data is split, users can run the prediction function and download the resulting predictions as a CSV file.

## Project Structure

The project is designed with modularity in mind, following an object-oriented programming (OOP) approach for scalability and clean code organization. Each component of the application, such as model selection, data handling, and prediction generation, is encapsulated within classes.

- **ModelSelector Class**: Manages the dropdown interface for selecting pre-trained machine learning models. This class will handle the user's input and ensure the selected model is stored for further processing.
  
- **Future Modules**:
  - A class to handle CSV file uploads and convert the data into a Pandas DataFrame.
  - A class or function to allow users to select features and target variables from the uploaded data.
  - A method to manage train-test split and prediction generation.

## Getting Started

To set up the project locally, follow these steps:

1. **Install Dependencies**  
   Ensure you have Python installed. You will need to install the necessary libraries using the following command:
   ```bash
   pip install streamlit pandas scikit-learn
