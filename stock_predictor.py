# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


stock_data = pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
print(stock_data.shape)
print(stock_data.sample(7))

stock_data.info()

stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data.info()

companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

amazon = stock_data[stock_data['Name'] == 'AMZN']
prediction_range = amazon.loc[(amazon['date'] > datetime(2013,1,1))
 & (amazon['date']<datetime(2018,1,1))]
plt.plot(amazon['date'],amazon['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("amazon Stock Prices")
plt.show()

close_data = amazon.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]
# prepare feature and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=10)

test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

initial_mse = np.mean(((predictions - y_test) ** 2))
initial_rmse = np.sqrt(initial_mse)

print("MSE", initial_mse)
print("RMSE", initial_rmse)

train = amazon[:training]
test = amazon[training:]
test['Predictions'] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train['date'], train['close'])
plt.plot(test['date'], test[['close', 'Predictions']])
plt.title('amazon Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])



# Tune the hyperparameters of the LSTM model to improve its performance.

## Identify hyperparameters to tune

"""
Determine which hyperparameters of the LSTM model (e.g., number of units in LSTM layers, dropout rate, number of dense layers, learning rate) will be tuned.

I want to review the existing LSTM model architecture and then identify which hyperparameters to tune based on common practices in tuning LSTM models for time series data.

 1. units in LSTM layers (64)
 2. dropout rate (0.5)
 3. number of dense layers (2 including the output layer) - though typically the output layer is fixed
 4. units in the dense layer (32 in the hidden dense layer)
 5. optimizer learning rate (optimizer is 'adam', learning rate is its default, could change this in the future)
"""


## Implement the tuning process

### Perform the hyperparameter tuning using Keras Tuner's RandomSearch and the identified hyperparameters.

### Define the model building function, instantiate the tuner, and start the search as outlined in the instructions.

amazon = stock_data[stock_data['Name'] == 'AMZN']
prediction_range = amazon.loc[(amazon['date'] > datetime(2013,1,1))
 & (amazon['date']<datetime(2018,1,1))]
plt.plot(amazon['date'],amazon['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Amazon Stock Prices")
plt.show()

import keras_tuner as kt

# Re-execute data preparation to ensure x_train and y_train are defined
close_data = amazon.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]

# prepare feature and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

def build_model(hp, input_shape):
    """Builds a Keras Sequential model for hyperparameter tuning."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
                                return_sequences=True,
                                input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)))
    model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)))
    model.add(tf.keras.layers.Dense(1)) # Output layer

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='mean_squared_error')

    return model

# Define the input shape based on x_train after re-creating it
input_shape = (x_train.shape[1], 1)

# Instantiate the RandomSearch tuner, passing the input_shape to the build_model function
tuner = kt.RandomSearch(
    lambda hp: build_model(hp, input_shape=input_shape),
    objective='loss',  # Optimize for training loss
    max_trials=10,     # Number of trials to run
    executions_per_trial=2, # Number of models to build and train for each trial
    directory='keras_tuner_dir', # Directory to save tuning results
    project_name='stock_prediction_tuning')

# Start the hyperparameter search
tuner.search(x_train, y_train, epochs=10)

"""
I loaded the data from the CSV file into a pandas DataFrame and displayed the first few rows and the columns and their data types in order understand the structure of the data visually.

"""

# Display the first 5 rows.
print(stock_data.head().to_markdown(index=False, numalign="left", stralign="left"))

# Print the column names and their data types.
print(stock_data.info())




## Select the best model

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
best_model = build_model(best_hps, input_shape)

# Train the best model on the entire training data
print("Training the best model...")
best_model.fit(x_train, y_train, epochs=50)       # Adjust the number of epochs



## Train and evaluate models

"""
Models have already been trained with different combinations of hyperparameters and evaluate their performance using appropriate metrics (e.g., RMSE, MSE).

Now I retrieved the best models found by the tuner and evaluated their performance on the test dataset. This allowed me to compare the performance of models with different hyperparameter combinations.
"""

# Get the top models from the tuner
best_models = tuner.get_best_models(num_models=5)

# Evaluate the best models on the test data
for i, model in enumerate(best_models):
    print(f"Evaluating best model {i+1}...")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"Model {i+1} Test Loss (MSE): {loss}")

    # Make predictions and calculate RMSE
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f"Model {i+1} Test RMSE: {rmse}")

"""
I evaluated the performance of the final model on the test dataset using appropriate metrics (e.g., RMSE, MSE) and compared it to the performance of the initial model.
"""

# Evaluate the final model
print("Evaluating the final model...")
loss = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Final Model Test Loss (MSE): {loss}")

# Make predictions and calculate RMSE
predictions = best_model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"Final Model Test RMSE: {rmse}\n")

# Compare with the initial model's performance
initial_mse = 769.9239578818352
initial_rmse = 27.747503633333128
print(f"\nInitial Model Test Loss (MSE): {initial_mse}")
print(f"Initial Model Test RMSE: {initial_rmse}")

# Plot the results
train = amazon[:training]
test = amazon[training:]
test['Predictions'] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train['date'], train['close'])
plt.plot(test['date'], test[['close', 'Predictions']])
plt.title('amazon Stock Close Price - Tuned Model')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()


# Transfer Learning
## Prepare a General Pre-training Dataset

"""
I selected a broader dataset of stocks from multiple companies to create a 'general' dataset for pre-training a base model on overall stock market patterns. This involved similar preprocessing steps as the single-stock data preparation.

The first step is to identify all unique stock names present in the `stock_data` DataFrame to iterate through them for creating the general dataset.
"""

unique_stocks = stock_data['Name'].unique()
print(f"Number of unique stocks: {len(unique_stocks)}")
print(f"First 5 unique stocks: {unique_stocks[:5]}")

"""
Now the unique stock names are identified, I iterated through each stock, extracted and preprocessed its closing prices, and then created the general training dataset using a sliding window approach.
"""

x_train_general = []
y_train_general = []

# Ensure the scaler is re-instantiated for each stock to avoid data leakage issues across stocks

for stock_name in unique_stocks:
    stock_df = stock_data[stock_data['Name'] == stock_name]

    # Ensure 'close' column exists and handle potential NaNs in 'close' before scaling
    stock_close_data = stock_df.filter(['close']).dropna()

    if len(stock_close_data) < 61: # Need at least 61 data points for a 60-step window
        continue # Skip stocks with insufficient data

    dataset_stock = stock_close_data.values

    # Apply MinMaxScaler
    scaler_stock = MinMaxScaler(feature_range=(0, 1))
    scaled_data_stock = scaler_stock.fit_transform(dataset_stock)

    # Create sequences for current stock
    for i in range(60, len(scaled_data_stock)):
        x_train_general.append(scaled_data_stock[i-60:i, 0])
        y_train_general.append(scaled_data_stock[i, 0])

# Convert to numpy arrays
x_train_general = np.array(x_train_general)
y_train_general = np.array(y_train_general)

# Reshape x_train_general for LSTM input
x_train_general = np.reshape(x_train_general, (x_train_general.shape[0], x_train_general.shape[1], 1))

print(f"Shape of x_train_general: {x_train_general.shape}")
print(f"Shape of y_train_general: {y_train_general.shape}")



## Define and Pre-train a Base Model
"""
I defined the base LSTM model architecture for pre-training, compile it, and train it on the `x_train_general` and `y_train_general` datasets to learn general stock market dynamics.
"""

pre_trained_model = keras.models.Sequential()
pre_trained_model.add(keras.layers.LSTM(units=64,
                                    return_sequences=True,
                                    input_shape=(x_train_general.shape[1], 1)))
pre_trained_model.add(keras.layers.LSTM(units=64))
pre_trained_model.add(keras.layers.Dense(32))
pre_trained_model.add(keras.layers.Dropout(0.5))
pre_trained_model.add(keras.layers.Dense(1))

pre_trained_model.compile(optimizer='adam', loss='mean_squared_error')

print("Pre-training the base model...")
history_pretrain = pre_trained_model.fit(x_train_general, y_train_general, epochs=5, batch_size=256)

print("Pre-training complete.")

pre_trained_model.save("general_model_epoch5_batchsize256.keras")

"""
I fine-tuned the pre-trained model using the prepared 'AMZN' specific stock data (x_fine_tune, y_fine_tune). After fine-tuning, I evaluated the model's performance on the 'AMZN' test data using RMSE and MSE, and then visualize the actual vs. predicted 'AMZN' stock prices. Finally, I summarized the performance of the fine-tuned model compared to previous models.
"""

## Load Pre-trained Model

fine_tune_model = tf.keras.models.load_model('general_model_epoch5_batchsize256.keras')
print("Pre-trained model loaded successfully.")

"""
I compiled the loaded model with a lower learning rate, a common practice in transfer learning for fine-tuning.
"""

fine_tune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
print("Fine-tune model compiled with Adam optimizer and a lower learning rate.")

"""
I prepared the Amazon-specific stock data for fine-tuning. This includes scaling the `amazon` close prices and creating sequences using a sliding window.
"""

fine_tune_stock_data = stock_data[stock_data['Name'] == 'AMZN']
fine_tune_close_data = fine_tune_stock_data.filter(['close'])
fine_tune_dataset = fine_tune_close_data.values

# Use the same training split as before for consistency
training_fine_tune = int(np.ceil(len(fine_tune_dataset) * .95))

scaler_fine_tune = MinMaxScaler(feature_range=(0, 1))
scaled_fine_tune_data = scaler_fine_tune.fit_transform(fine_tune_dataset)

train_fine_tune_data = scaled_fine_tune_data[0:int(training_fine_tune), :]

x_fine_tune = []
y_fine_tune = []

for i in range(60, len(train_fine_tune_data)):
    x_fine_tune.append(train_fine_tune_data[i-60:i, 0])
    y_fine_tune.append(train_fine_tune_data[i, 0])

x_fine_tune, y_fine_tune = np.array(x_fine_tune), np.array(y_fine_tune)
x_fine_tune = np.reshape(x_fine_tune, (x_fine_tune.shape[0], x_fine_tune.shape[1], 1))

print(f"Shape of x_fine_tune: {x_fine_tune.shape}")
print(f"Shape of y_fine_tune: {y_fine_tune.shape}")



"""
I fine-tuned the model using the `x_fine_tune` and `y_fine_tune` datasets.
"""

print("Fine-tuning the model...")
history_finetune = fine_tune_model.fit(x_fine_tune, y_fine_tune, epochs=50, batch_size=32)
print("Fine-tuning complete.")

"""
I then prepared the test data for evaluation, made predictions with the fine-tuned model, calculated the RMSE and MSE, and compared the metrics against the initial model's performance to assess the impact of transfer learning.
"""

test_fine_tune_data = scaled_fine_tune_data[training_fine_tune - 60:, :]
x_test_fine_tune = []
y_test_fine_tune = fine_tune_dataset[training_fine_tune:, :]

for i in range(60, len(test_fine_tune_data)):
    x_test_fine_tune.append(test_fine_tune_data[i-60:i, 0])

x_test_fine_tune = np.array(x_test_fine_tune)
x_test_fine_tune = np.reshape(x_test_fine_tune, (x_test_fine_tune.shape[0], x_test_fine_tune.shape[1], 1))

print("Evaluating the fine-tuned model...")
predictions_finetuned = fine_tune_model.predict(x_test_fine_tune)
predictions_finetuned = scaler_fine_tune.inverse_transform(predictions_finetuned)

mse_finetuned = np.mean(((predictions_finetuned - y_test_fine_tune) ** 2))
rmse_finetuned = np.sqrt(mse_finetuned)

print(f"Fine-tuned Model Test Loss (MSE): {mse_finetuned}")
print(f"Fine-tuned Model Test RMSE: {rmse_finetuned}")


print(f"\nInitial Model Test Loss (MSE): {initial_mse}")
print(f"Initial Model Test RMSE: {initial_rmse}")

print(f"\nTuned Model Test Loss (MSE): {loss}") # loss from previous best_model evaluation
print(f"Tuned Model Test RMSE: {rmse}") # rmse from previous best_model evaluation

"""
I plotted the actual vs. predicted stock prices using the fine-tuned model. 
This created a plot similar to the initial model's visualization, showing the training data, the actual test data, and the predictions made by the fine-tuned model.
"""

amazon = stock_data[stock_data['Name'] == 'AMZN']
prediction_range = amazon.loc[(amazon['date'] > datetime(2013,1,1))
 & (amazon['date']<datetime(2018,12,31))]
plt.plot(amazon['date'],amazon['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Amazon Stock Prices")
plt.show()

train_finetuned = amazon[:training_fine_tune]
test_finetuned = amazon[training_fine_tune:]
test_finetuned['Predictions'] = predictions_finetuned

plt.figure(figsize=(10, 8))
plt.plot(train_finetuned['date'], train_finetuned['close'])
plt.plot(test_finetuned['date'], test_finetuned[['close', 'Predictions']])
plt.title('Amazon Stock Close Price - Fine-tuned Model')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()

