import pandas as pd
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


def lstm_model():
    # Load the dataset
    data = pd.read_csv('C:\\Users\\KIIT\\Desktop\\AD\\Lab3\\data\\stock_data.csv')  
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Normalize the Close prices
    scaler = MinMaxScaler()
    data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Split data into training and testing
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Prepare data for LSTM
    look_back = 10
    train_scaled = train_data['Close'].values.reshape(-1, 1)
    test_scaled = test_data['Close'].values.reshape(-1, 1)

    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Visualization: Actual vs Predicted Close Prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Prices", color='blue', alpha=0.6)
    plt.plot(predictions, label="Predicted Prices", color='red', alpha=0.6)
    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")  # Save the graph as an image
    plt.show()

    # Visualization: Training Loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label="Training Loss", color='green')
    plt.title("Model Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_loss.png")  # Save the graph as an image
    plt.show()

    return {
        "MSE": mse,
        "R2": r2,
        "Predictions": predictions.flatten().tolist()
    }


if __name__ == '__main__':
    results = lstm_model()
    print("LSTM Results:")
    print(f"Mean Squared Error: {results['MSE']}")
    print(f"R2 Score: {results['R2']}")


# import numpy as np
# import matplotlib.pyplot as plt
# from keras.api.models import Sequential
# from keras.api.layers import Dense, LSTM
# from sklearn.metrics import mean_squared_error, r2_score

# def lstm_model():
#     # Dummy time-series data
#     data = np.sin(np.linspace(0, 50, 100))
#     look_back = 5

#     # Prepare LSTM dataset
#     X, y = [], []
#     for i in range(len(data) - look_back):
#         X.append(data[i:i + look_back])
#         y.append(data[i + look_back])

#     X, y = np.array(X), np.array(y)
#     X = X.reshape(X.shape[0], X.shape[1], 1)

#     # Build LSTM model
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=False, input_shape=(look_back, 1)))
#     model.add(Dense(1))
#     model.compile(optimizer="adam", loss="mean_squared_error")
#     model.fit(X, y, epochs=5, verbose=0)

#     # Predictions
#     y_pred = model.predict(X)
#     mse = mean_squared_error(y, y_pred)
#     r2 = r2_score(y, y_pred)

#     # Save graph
#     plt.figure()
#     plt.plot(range(len(y)), y, label="Actual", marker="o")
#     plt.plot(range(len(y_pred)), y_pred, label="Predicted", linestyle="--")
#     plt.title("LSTM Model")
#     plt.legend()
#     plt.savefig("backend/static/graphs/lstm_graph.png")
#     plt.close()

#     return {"MSE": mse, "R2": r2}
