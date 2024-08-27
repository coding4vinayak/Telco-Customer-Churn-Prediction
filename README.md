

# Telco Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company. By analyzing historical customer data, the project aims to identify factors that contribute to customer attrition and build a predictive model to help the company retain valuable customers.

## Features

- **Churn Prediction**: Predict the likelihood of a customer churning based on historical data.
- **Data Analysis**: Explore and analyze customer data to identify key factors influencing churn.
- **Predictive Modeling**: Build and evaluate machine learning models for churn prediction.
- **Visualization**: Visualize customer churn trends, model performance, and key factors.
- **Web Interface**: Provide a Flask-based web interface to input customer data and view churn predictions.

## Prerequisites

- Python 3.x
- Flask (for web application interface)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook (for exploratory data analysis)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/telco-customer-churn.git
   cd telco-customer-churn
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Set up environment variables if needed.

## Usage

### Data Analysis

1. **Load and Explore Data**:
   - Load customer data and perform exploratory data analysis to understand churn patterns.
   - Analyze distributions of features such as customer tenure, contract type, payment method, and service usage.

2. **Preprocess Data**:
   - Clean the data by handling missing values, encoding categorical variables, and scaling numerical features.

### Predictive Modeling

1. **Train Models**:
   - Implement and train various classification algorithms (e.g., Logistic Regression, Random Forest, Gradient Boosting) to predict customer churn.

2. **Evaluate Models**:
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

3. **Optimize Models**:
   - Tune hyperparameters to improve model performance and avoid overfitting.

### Running the Flask Application

1. Start the Flask server:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000` to access the web interface for inputting customer data and viewing churn predictions.

## Code Overview

- **app.py**: Main Flask application script.
  - **index()**: Renders the main page.
  - **predict()**: Handles prediction of customer churn based on user input.

- **model.py**: Contains machine learning model implementation and evaluation.
  - **train_models()**: Trains different classification models for churn prediction.
  - **evaluate_models()**: Evaluates model performance.
  - **predict_churn()**: Predicts churn likelihood based on input features.

- **data_preprocessing.py**: Handles data loading, cleaning, and preprocessing.

- **requirements.txt**: Lists the Python packages required for the project.

- **notebooks/**: Contains Jupyter Notebooks for exploratory data analysis and model training.

## Configuration

- **MODEL_PATH**: Path to the trained model file if loading from disk.
- **FLASK_APP_PORT**: Port for the Flask application (default: 5000).

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

## More Projects

Check out more of my projects at [coding4vinayak](https://vinayakss.vercel.app/).

