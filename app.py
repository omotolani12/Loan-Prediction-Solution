import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer  # Import imputer
import pickle

# Load the saved variables
with open('training_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)

# Load your pre-trained model
rf_model = RandomForestClassifier()  # You need to replace this with your actual trained model
rf_model.fit(X_train, y_train)  # Train your model with your actual training data

# Define the app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Loan Approval Predictor"),
    html.Label("Enter ID:"),
    dcc.Input(id='input-id', type='text', placeholder='Enter ID'),

    html.Label("Select Gender:"),
    dcc.Dropdown(
        id='input-gender',
        options=[
            {'label': 'Male', 'value': 'Male'},
            {'label': 'Female', 'value': 'Female'},
            # Add other gender options as needed
        ],
        placeholder='Select Gender'
    ),

    html.Label("Enter Age:"),
    dcc.Input(id='input-age', type='number', placeholder='Enter Age'),

    html.Label("Enter Income:"),
    dcc.Input(id='input-income', type='number', placeholder='Enter Income'),

    html.Label("Select Region:"),
    dcc.Dropdown(
        id='input-region',
        options=[
            {'label': 'North', 'value': 'North'},
            {'label': 'south', 'value': 'south'},
            {'label': 'central', 'value': 'central'},
            {'label': 'east', 'value': 'east'},
            # Add other region options as needed
        ],
        placeholder='Select Region'
    ),

    html.Label("Enter Loan Amount:"),
    dcc.Input(id='input-loan-amount', type='number', placeholder='Enter Loan Amount'),

    html.Label("Select Loan Purpose:"),
    dcc.Dropdown(
        id='input-loan-purpose',
        options=[
            {'label': 'p1', 'value': 'p1'},
            {'label': 'p2', 'value': 'p2'},
            {'label': 'p3', 'value': 'p3'},
            {'label': 'p4', 'value': 'p4'},
            # Add other loan purpose options as needed
        ],
        placeholder='Select Loan Purpose'
    ),

    html.Label("Select Loan Type:"),
    dcc.Dropdown(
        id='input-loan-type',
        options=[
            {'label': 'Type A', 'value': 'type1'},
            {'label': 'Type B', 'value': 'type2'},
            # Add other loan type options as needed
        ],
        placeholder='Select Loan Type'
    ),

    html.Label("Select Use Type:"),
    dcc.Dropdown(
        id='input-use-type',
        options=[
            {'label': 'Business/Commercial', 'value': 'b/c'},
            {'label': 'N/A', 'value': 'nob/c'},
            # Add other options as needed
        ],
        placeholder='Select Use Type'
    ),

    html.Label("Select Credit Type:"),
    dcc.Dropdown(
        id='input-credit-type',
        options=[
            {'label': 'CIB', 'value': 'CIB'},
            {'label': 'CRIF', 'value': 'CRIF'},
            {'label': 'EQUI', 'value': 'EQUI'},
            {'label': 'EXP', 'value': 'EXP'},
            # Add other options as needed
        ],
        placeholder='Select credit Type'
    ),

    html.Label("Select Loan Limit:"),
    dcc.Dropdown(
        id='input-loan-limit',
        options=[
            {'label': 'CF', 'value': 'CF'},
            {'label': 'NCF', 'value': 'NCF'},
            # Add other options as needed
        ],
        placeholder='Select Loan Limit'
    ),

    html.Label("Select Credit Worthiness:"),
    dcc.Dropdown(
        id='input-credit-worthiness',
        options=[
            {'label': 'Level 1', 'value': 'l1'},
            {'label': 'Level 2', 'value': 'l2'},
            # Add other options as needed
        ],
        placeholder='Select Credit Worthiness'
    ),

    html.Label("Enter Credit Score:"),
    dcc.Input(id='input-credit-score', type='number', placeholder='Enter Credit Score'),

    html.Label("Enter Property Value:"),
    dcc.Input(id='input-property-value', type='number', placeholder='Enter Property Value'),

    # Add more input fields as needed...

    html.Button('Predict', id='predict-button'),
    html.Div(id='output-prediction')
])

# Define callback to update output based on input
@app.callback(
    Output('output-prediction', 'children'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('input-gender', 'value'),
     dash.dependencies.State('input-age', 'value'),
     dash.dependencies.State('input-income', 'value'),
     dash.dependencies.State('input-region', 'value'),
     dash.dependencies.State('input-loan-amount', 'value'),
     dash.dependencies.State('input-loan-purpose', 'value'),
     dash.dependencies.State('input-loan-type', 'value'),
     dash.dependencies.State('input-use-type', 'value'),
     dash.dependencies.State('input-credit-type', 'value'),
     dash.dependencies.State('input-loan-limit', 'value'),
     dash.dependencies.State('input-credit-worthiness', 'value'),
     dash.dependencies.State('input-credit-score', 'value'),
     dash.dependencies.State('input-property-value', 'value'),
     # Add other State components for additional input fields
     # ...
     ]
)
def update_prediction(n_clicks, input_gender, input_age, input_income, input_region, input_loan_amount, input_loan_purpose,
                      input_loan_type, input_use_type, input_credit_type, input_loan_limit, input_credit_worthiness, input_credit_score, input_property_value):
    # Define relevant columns for prediction
    relevant_columns = [
        'loan_limit', 'Gender', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'business_or_commercial', 'loan_amount',
        'property_value', 'income', 'credit_type', 'Credit_Score',
        'age', 'Region', 'risk_category'
    ]

    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'Gender': [input_gender],
        'age': [input_age],
        'income': [input_income],
        'Region': [input_region],
        'loan_amount': [input_loan_amount],
        'loan_purpose': [input_loan_purpose],
        'loan_type': [input_loan_type],
        'business_or_commercial': [input_use_type],
        'credit_type': [input_credit_type],
        'loan_limit': [input_loan_limit],
        'Credit_Worthiness': [input_credit_worthiness],
        'Credit_Score': [input_credit_score],
        'property_value': [input_property_value],
        # Add more columns as needed...
    }, columns=relevant_columns)

    # Apply label encoding to categorical columns
    label_encoder = LabelEncoder()
    for col in ['Gender', 'Region', 'loan_purpose', 'loan_type', 'business_or_commercial', 'Credit_Worthiness', 'credit_type']:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    # Drop rows with missing values
    input_data = input_data.dropna()

    # Check if DataFrame is empty
    if input_data.empty:
        return "No predictions can be made. Please provide valid input."

    # Make predictions using your pre-trained model
    prediction = rf_model.predict(input_data)

    return f"Loan Approval Prediction: {'Approved' if prediction == 1 else 'Not Approved'}"

# ...

if __name__ == '__main__':
    app.run_server(debug=True)



