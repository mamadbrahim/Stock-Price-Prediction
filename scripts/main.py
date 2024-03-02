import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from functions import *


path_dataset = "NSE-Tata-Global-Beverages-Limited.csv"

# Load your dataset
data = pd.read_csv(path_dataset)

# Choose the column to predict
target_column = 'Close'

# Extract the target variable
target = data[target_column].values.reshape(-1, 1)

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler.fit_transform(target)

# Choose the sequence length (number of time steps to look back)
sequence_length = 10

# Create sequences and split data into train and test sets
sequences = create_sequences(target_scaled, sequence_length)
X, y = sequences[:, :-1], sequences[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions to original scale
predictions_inverse = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test_inverse, predictions_inverse)
print(f'Mean Squared Error: {mse}')



title='Actual vs. Predicted Stock Prices' ; xlabel='Time' ;ylabel='Stock Price'
actual = y_test_inverse ; predicted = predictions_inverse
# Plot 
plot_actual_vs_predicted(actual, predicted, title, xlabel, ylabel)

# if you are sutisfied we the model you can uncomment the next line to save it
# model.save("saved_model.h5")