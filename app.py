from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO
from flasgger import Swagger

app = Flask(__name__)

CSV_FILE = "CAR DETAILS FROM CAR DEKHO.csv"

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Car Price Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cars.db'
db = SQLAlchemy(app)

# Define a database model
class Car(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    selling_price = db.Column(db.Float, nullable=False)
    year = db.Column(db.Integer, nullable=False)
    km_driven = db.Column(db.Integer, nullable=False)
    fuel = db.Column(db.String(20), nullable=False)
    seller_type = db.Column(db.String(20), nullable=False)
    transmission = db.Column(db.String(20), nullable=False)
    owner = db.Column(db.String(20), nullable=False)

# Create the database
with app.app_context():
    db.create_all()

def preprocess_data(df):
    # Drop the car name since it's not useful for predictions
    df = df.drop(columns=['name'])

    # Handle missing values: Fill numerical columns with median values
    df['year'] = df['year'].fillna(df['year'].median())
    df['km_driven'] = df['km_driven'].fillna(df['km_driven'].median())

    # Fill missing categorical values with the most frequent value
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
    encoded_features = encoder.fit_transform(df[categorical_cols])

    # Create a DataFrame for the one-hot encoded categorical variables
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate with encoded features
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Normalize numerical columns
    scaler = StandardScaler()
    df[['year', 'km_driven']] = scaler.fit_transform(df[['year', 'km_driven']])

    # Separate features (X) and target (y)
    X = df.drop(columns=['selling_price'])  # Features
    y = df['selling_price']  # Target variable

    return X, y, encoder

# Global variables for model and encoder
model = None
encoder = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the Car Dekho dataset, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model, encoder

    # Step 1: Check if the dataset exists
    if not os.path.exists(CSV_FILE):
        return jsonify({"error": "Dataset file not found."}), 400

    # Step 2: Load data into pandas
    df = pd.read_csv(CSV_FILE)

    # Step 3: Clear the database
    db.session.query(Car).delete()

    # Step 4: Process data and insert it into the database
    df = df.dropna()  # Drop rows with missing values

    for _, row in df.iterrows():
        new_car = Car(
            selling_price=row['selling_price'],
            year=row['year'],
            km_driven=row['km_driven'],
            fuel=row['fuel'],
            seller_type=row['seller_type'],
            transmission=row['transmission'],
            owner=row['owner']
        )
        db.session.add(new_car)
    db.session.commit()

    # Step 5: Preprocess and train model
    X, y, encoder = preprocess_data(df)
    model = LinearRegression()
    model.fit(X, y)

    # Step 6: Generate summary statistics
    summary = {
        'total_cars': len(df),
        'average_price': df['selling_price'].mean(),
        'min_price': df['selling_price'].min(),
        'max_price': df['selling_price'].max(),
        'average_year': df['year'].mean(),
        'average_km_driven': df['km_driven'].mean(),
        'top_fuel_types': df['fuel'].value_counts().head().to_dict()
    }

    return jsonify(summary)
@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the selling price of a car
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            year:
              type: integer
            km_driven:
              type: integer
            fuel:
              type: string
            seller_type:
              type: string
            transmission:
              type: string
            owner:
              type: string
    responses:
      200:
        description: Predicted car selling price
    '''
    global model, encoder  # Ensure that the encoder and model are available for prediction
    
    # Check if the model and encoder are initialized
    if model is None or encoder is None:
        return jsonify({"error": "The data has not been loaded. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        # Extract numerical inputs
        year = pd.to_numeric(data.get('year'), errors='coerce')
        km_driven = pd.to_numeric(data.get('km_driven'), errors='coerce')

        # Extract categorical inputs
        fuel = data.get('fuel')
        seller_type = data.get('seller_type')
        transmission = data.get('transmission')
        owner = data.get('owner')

        if None in [year, km_driven, fuel, seller_type, transmission, owner]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400

        # Check for NaN values in numerical inputs
        if pd.isna(year) or pd.isna(km_driven):
            return jsonify({"error": "Invalid numeric values for year or km_driven"}), 400

        # One-hot encode categorical variables
        categorical_values = [[fuel, seller_type, transmission, owner]]
        categorical_encoded = encoder.transform(categorical_values)

        # Normalize numerical features
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(np.array([[year, km_driven]]))
        
        # Combine numerical and encoded categorical features
        input_data = np.concatenate((numerical_features[0], categorical_encoded[0]))
        input_data = input_data.reshape(1, -1)

        # Predict the selling price
        predicted_price = model.predict(input_data)[0]

        return jsonify({"predicted_selling_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
