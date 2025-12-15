# src/web_interface/app.py
from flask import Flask, render_template, jsonify
from database.db_manager import TrafficDatabase
from ml_models.traffic_predictor import TrafficPredictor
import threading
import time

app = Flask(__name__)

# Initialize components
db = TrafficDatabase()
predictor = TrafficPredictor('models/traffic_model.pt')

# Background thread: continuously make predictions
def prediction_loop():
    while True:
        links = db.get_all_links()
        for link in links:
            prediction = predictor.predict_next_frame(link, db)
            db.store_prediction(prediction)  # Save for comparison
        time.sleep(5)  # Predict every 5 seconds

# Start background thread
prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
prediction_thread.start()

# Web routes
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/current')
def api_current():
    return jsonify(db.get_current_traffic())

@app.route('/api/predicted')
def api_predicted():
    return jsonify(db.get_latest_predictions())

if __name__ == '__main__':
    app.run(debug=True, port=5000)
