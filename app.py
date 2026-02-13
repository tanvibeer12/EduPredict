from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from model import StudentPerformanceModel
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize the ML model
model = StudentPerformanceModel()

# Train the model on startup (in production, load a pre-trained model)
print("Training model...")
model.train_model()
print("Model training completed!")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    Expected JSON format:
    {
        "study_hours": 5.0,
        "attendance_rate": 85.0,
        "previous_grade": 75.0,
        "assignments_completed": 80.0,
        "extracurriculars": 5.0,
        "sleep_hours": 7.0
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input
        required_fields = [
            'study_hours', 
            'attendance_rate', 
            'previous_grade',
            'assignments_completed', 
            'extracurriculars', 
            'sleep_hours'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract features
        features = np.array([[
            data['study_hours'],
            data['attendance_rate'],
            data['previous_grade'],
            data['assignments_completed'],
            data['extracurriculars'],
            data['sleep_hours']
        ]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get detailed results
        results = model.get_prediction_details(features[0])
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def statistics():
    """
    Endpoint for getting model statistics
    """
    try:
        stats = model.get_model_statistics()
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model.is_trained
    })

if __name__ == '__main__':
    # Run the Flask app
    # In production, use a proper WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)
