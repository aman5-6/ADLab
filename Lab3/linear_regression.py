
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def linear_regression_model():
    # File path for stock data
    file_path = 'C:\\Users\\KIIT\\Desktop\\AD\\Lab3\\data\\stock_data.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load data
    data = pd.read_csv(file_path)

    # Ensure necessary columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Feature selection (X) and target variable (y)
    X = data[['Open', 'High', 'Low']]  # Using Open, High, and Low as features
    y = data['Close']  # Using Close price as the target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model initialization
    model = LinearRegression()

    # Training the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  # Accuracy as a percentage

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual Values", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted Values", alpha=0.6)
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Data Points")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('output_visualization.png')  # Save the graph as an image
    plt.show()

    # Return results
    results = {
        "message": "Linear regression model executed successfully.",
        "model_coefficients": model.coef_.tolist(),  # Coefficients of the features
        "model_intercept": model.intercept_,  # Intercept of the regression line
        "mean_squared_error": mse,
        "r2_score": r2,
        "accuracy": accuracy,
        "sample_predictions": {
            "actual": y_test.tolist()[:5],  # First 5 actual values
            "predicted": y_pred.tolist()[:5]  # First 5 predicted values
        }
    }

    return results


if __name__ == '__main__':
    try:
        results = linear_regression_model()
        print("Linear Regression Results:")
        print(f"Mean Squared Error: {results['mean_squared_error']}")
        print(f"R2 Score: {results['r2_score']}")
        print(f"Accuracy: {results['accuracy']}%")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
