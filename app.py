from flask import Flask, request, jsonify, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the uploaded file
        file = request.files['file']
        data = pd.read_csv(file)

        # Initialize prediction pipeline
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(data)

        # Add predictions to the DataFrame
        data['Predictions'] = predictions

        # Return the result as HTML
        return data.to_html()
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True,port=5001)
