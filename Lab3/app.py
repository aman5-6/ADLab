from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
from linear_regression import linear_regression_model  # Import your Linear Regression code
from lstm_model import lstm_model  # Import your LSTM code
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS to handle frontend requests

# Function to generate a bar chart for model accuracies
def create_accuracy_chart(lr_accuracy, lstm_accuracy):
    try:
        # Data for the chart
        models = ['Linear Regression', 'LSTM']
        accuracies = [lr_accuracy, lstm_accuracy]

        # Create a bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(models, accuracies, color=['blue', 'green'])
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)

        # Save the chart
        chart_path = os.path.join('static', 'graphs', 'accuracy_comparison.png')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        plt.savefig(chart_path)
        plt.close()

        return chart_path
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')  # No absolute path needed; Flask looks in 'templates'

@app.route('/compare-models', methods=['GET'])
def compare_models():
    try:
        # Run both models and get their accuracies
        lr_results = linear_regression_model()
        lstm_results = lstm_model()

        # Calculate accuracy percentages
        lr_accuracy = lr_results["accuracy"] * 100  # Assuming accuracy is already in percentage form
        lstm_accuracy = lstm_results["R2"] * 100  # Assuming R2 is converted to percentage form

        # Generate the comparison chart
        chart_path = create_accuracy_chart(lr_accuracy, lstm_accuracy)

        if not chart_path:
            return jsonify({"error": "Error generating comparison chart"}), 500

        # Respond with the chart's URL
        response = {
            "chart_url": f"/static/graphs/accuracy_comparison.png"
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())  # Log full traceback
        return jsonify({"error": str(e)}), 500

# Route to serve static files (like graphs)
@app.route('/static/<path:filename>', methods=['GET'])
def serve_static(filename):
    static_dir = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_dir, filename)

if __name__ == '__main__':
    # Ensure the static directory exists
    os.makedirs('static/graphs', exist_ok=True)
    app.run(debug=True)
