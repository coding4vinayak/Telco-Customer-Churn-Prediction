from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained scaler and churn model
scaler = joblib.load('app/model/scaler.pkl')
churn_model = joblib.load('app/model/churn_model.pkl')

# Define a function to map categorical input to numerical values
def preprocess_input(data):
    # Map categorical features to numerical values
    gender_map = {'Male': 1, 'Female': 0}
    partner_map = {'Yes': 1, 'No': 0}
    dependents_map = {'Yes': 1, 'No': 0}
    phone_service_map = {'Yes': 1, 'No': 0}
    multiple_lines_map = {'Yes': 1, 'No': 0, 'No phone service': 2}
    internet_service_map = {'DSL': 1, 'Fiber optic': 2, 'No': 0}
    online_security_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    online_backup_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    device_protection_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    tech_support_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    streaming_tv_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    streaming_movies_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paperless_billing_map = {'Yes': 1, 'No': 0}
    payment_method_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

    # Map data for each feature
    data['gender'] = gender_map[data['gender']]
    data['Partner'] = partner_map[data['Partner']]
    data['Dependents'] = dependents_map[data['Dependents']]
    data['PhoneService'] = phone_service_map[data['PhoneService']]
    data['MultipleLines'] = multiple_lines_map[data['MultipleLines']]
    data['InternetService'] = internet_service_map[data['InternetService']]
    data['OnlineSecurity'] = online_security_map[data['OnlineSecurity']]
    data['OnlineBackup'] = online_backup_map[data['OnlineBackup']]
    data['DeviceProtection'] = device_protection_map[data['DeviceProtection']]
    data['TechSupport'] = tech_support_map[data['TechSupport']]
    data['StreamingTV'] = streaming_tv_map[data['StreamingTV']]
    data['StreamingMovies'] = streaming_movies_map[data['StreamingMovies']]
    data['Contract'] = contract_map[data['Contract']]
    data['PaperlessBilling'] = paperless_billing_map[data['PaperlessBilling']]
    data['PaymentMethod'] = payment_method_map[data['PaymentMethod']]

    # Convert the data to a numpy array and reshape it for scaling
    input_array = np.array(list(data.values())).reshape(1, -1)
    
    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(input_array)

    return scaled_data

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get the input data from the form
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': request.form['SeniorCitizen'],
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': request.form['tenure'],
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': request.form['MonthlyCharges'],
            'TotalCharges': request.form['TotalCharges']
        }

        # Preprocess the input data (including scaling)
        processed_data = preprocess_input(input_data)

        # Make a prediction using the churn model
        prediction = churn_model.predict(processed_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
