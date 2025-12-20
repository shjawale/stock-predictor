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

"""
##Stock Predictor"""

stock_data = pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
print(stock_data.shape)
print(stock_data.sample(7))

stock_data.info()

stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data.info()

companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

amazon = stock_data[stock_data['Name'] == 'AAPL']
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

"""
# Tune the hyperparameters of the LSTM model to improve its performance.

## Identify hyperparameters to tune

Determine which hyperparameters of the LSTM model (e.g., number of units in LSTM layers, dropout rate, number of dense layers, learning rate) will be tuned.

I want to review the existing LSTM model architecture and then identify which hyperparameters to tune based on common practices in tuning LSTM models for time series data.
"""

# Based on the architecture, the hyperparameters to tune are:
# 1. units in LSTM layers (64)
# 2. dropout rate (0.5)
# 3. number of dense layers (2 including the output layer) - though typically the output layer is fixed
# 4. units in the dense layer (32 in the hidden dense layer)
# 5. optimizer learning rate (optimizer is 'adam', learning rate is its default, could change this in the future)


"""
## Choose a tuning method

### Select a method for hyperparameter tuning, such as Grid Search, Random Search, or using a hyperparameter tuning library like Keras Tuner or Optuna.

"""



"""
## Implement the tuning process

### Perform the hyperparameter tuning using Keras Tuner's RandomSearch and the identified hyperparameters.

"""

!pip install keras-tuner

"""
### Define the model building function, instantiate the tuner, and start the search as outlined in the instructions.


"""

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
The first step is to load the data from the CSV file into a pandas DataFrame and display the first few rows and the columns and their data types to understand the structure of the data.

"""

# Display the first 5 rows.
print(stock_data.head().to_markdown(index=False, numalign="left", stralign="left"))

# Print the column names and their data types.
print(stock_data.info())

"""
## Retrain the model with the best hyperparameters

### Train a final model using the entire training dataset and the best hyperparameters found during tuning.

## Select the best model

### Choose the model with the best performance on the evaluation metric.
"""

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
best_model = build_model(best_hps, input_shape)

# Train the best model on the entire training data
print("Training the best model...")
best_model.fit(x_train, y_train, epochs=50)       # Adjust the number of epochs

"""
## Train and evaluate models

Train models with different combinations of hyperparameters and evaluate their performance using appropriate metrics (e.g., RMSE, MSE).

Retrieve the best models found by the tuner and evaluate their performance on the test dataset. This will allow us to compare the performance of models with different hyperparameter combinations.
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
## Evaluate the final model

Evaluate the performance of the final model on the test dataset using appropriate metrics (e.g., RMSE, MSE) and compare it to the performance of the initial model.
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

"""
# Transfer Learning

## Prepare a General Pre-training Dataset

Select a broader dataset (e.g., a collection of multiple stocks from `stock_data`) to create a 'general' dataset for pre-training a base model on overall stock market patterns. This will involve similar preprocessing steps as the single-stock data preparation.

The first step is to identify all unique stock names present in the `stock_data` DataFrame to iterate through them for creating the general dataset.
"""

unique_stocks = stock_data['Name'].unique()
print(f"Number of unique stocks: {len(unique_stocks)}")
print(f"First 5 unique stocks: {unique_stocks[:5]}")

"""
Now that the unique stock names are identified, the next step is to iterate through each stock, extract and preprocess its closing prices, and then create the general training dataset using a sliding window approach, as described in the instructions.

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

"""
## Define and Pre-train a Base Model

Define the base LSTM model architecture for pre-training, compile it, and train it on the `x_train_general` and `y_train_general` datasets to learn general stock market dynamics.
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
# Fine-tune pre-trained model
Fine-tune the pre-trained model using the prepared 'AMZN' specific stock data (x_fine_tune, y_fine_tune). After fine-tuning, evaluate the model's performance on the 'AMZN' test data using RMSE and MSE, and then visualize the actual vs. predicted 'AMZN' stock prices. Finally, summarize the performance of the fine-tuned model compared to previous models.

## Load Pre-trained Model
"""

fine_tune_model = tf.keras.models.load_model('general_model_epoch5_batchsize256.keras')
print("Pre-trained model loaded successfully.")

"""
Compile the loaded model with a lower learning rate, a common practice in transfer learning for fine-tuning.

"""

fine_tune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
print("Fine-tune model compiled with Adam optimizer and a lower learning rate.")

"""
Prepare the Amazon-specific stock data for fine-tuning. This includes scaling the `amazon` close prices and creating sequences using a sliding window.

"""

fine_tune_stock_data = stock_data[stock_data['Name'] == 'GOOGL']
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
Fine-tune the model using the `x_fine_tune` and `y_fine_tune` datasets.

"""

print("Fine-tuning the model...")
history_finetune = fine_tune_model.fit(x_fine_tune, y_fine_tune, epochs=50, batch_size=32)
print("Fine-tuning complete.")

"""
Prepare the test data for evaluation, make predictions with the fine-tuned model, calculate the RMSE and MSE, and compare these metrics against the initial model's performance to assess the impact of transfer learning.

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
Plot the actual vs. predicted stock prices using the fine-tuned model. This step will create a plot similar to the initial model's visualization, showing the training data, the actual test data, and the predictions made by the fine-tuned model.

"""

google = stock_data[stock_data['Name'] == 'GOOGL']
prediction_range = google.loc[(google['date'] > datetime(2013,1,1))
 & (google['date']<datetime(2018,12,31))]
plt.plot(google['date'],amazon['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Google Stock Prices")
plt.show()

train_finetuned = google[:training_fine_tune]
test_finetuned = google[training_fine_tune:]
test_finetuned['Predictions'] = predictions_finetuned

plt.figure(figsize=(10, 8))
plt.plot(train_finetuned['date'], train_finetuned['close'])
plt.plot(test_finetuned['date'], test_finetuned[['close', 'Predictions']])
plt.title('Google Stock Close Price - Fine-tuned Model')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()

"""
### Summary of Model Performance

Let's compare the performance metrics (MSE and RMSE) for the initial model, the hyperparameter-tuned model, and the fine-tuned model with transfer learning.

   Initial Model (untuned):
       MSE: 51.18
       RMSE: 7.15

   Hyperparameter-Tuned Model (without transfer learning):**
       MSE: 29274.66
       RMSE: 1211.76

   Fine-tuned Model (with transfer learning):**
       MSE: 599.99
       RMSE: 24.49

Analysis:

The initial model showed the best performance with the lowest MSE and RMSE.  Hyperparameter tuning and transfer learning are generally expected to improve model performance, so this is unusual.

The hyperparameter-tuned model performed significantly worse than the initial model. This suggests that the hyperparameter search space or the number of trials was insufficient, or that the chosen objective, training loss, during tuning did not correlate well with generalization performance on the test set. It's also possible that the nunmber of executions per trial, 2, might not be enough to get a stable estimate of performance for each hyperparameter combination. Additionally, the hyperparameters found might have led to overfitting to the training data. The large increase in MSE and RMSE indicates a significant degradation in predictive accuracy.

The fine-tuned model using transfer learning also performed significantly worse than the initial model but considerably better than the hyperparameter-tuned model. While transfer learning generally helps, in this specific instance, the pre-trained model might not have captured market dynamics relevant to the 'AMZN' stock in a way that benefits fine-tuning. The pre-training dataset might have been too diverse, or the 'AMZN' stock's behavior is too distinct from the general market trends learned by the pre-trained model. It's also possible that the additional 10 epochs for fine-tuning were not enough, or that a different fine-tuning strategy (e.g., freezing some layers) would be more beneficial.

Conclusion:

Based on these results, the initial model performed best for predicting Amazon stock prices. This outcome highlights that advanced techniques like hyperparameter tuning and transfer learning are not guaranteed to improve performance and heavily depend on proper implementation, data characteristics, and problem domain. Further investigation into the hyperparameter tuning process (e.g., wider search space, more epochs per trial, different objective function) and the transfer learning setup (e.g., different pre-training data, different fine-tuning approach) would be necessary to potentially surpass the initial model's performance.
"""



"""
# Neflix stock data
"""

netflix_stock_data = pd.read_csv('NFLX.csv', delimiter=',', on_bad_lines='skip')
print(netflix_stock_data.info())
print(netflix_stock_data.shape)
netflix_stock_data.head()

"""
### Prepare Netflix Data

Convert the 'Date' column to datetime objects, extract the 'Close' prices, and then split the data into training and testing sets based on a 95/5 ratio.
"""

netflix_stock_data['Date'] = pd.to_datetime(netflix_stock_data['Date'])
netflix_close_data = netflix_stock_data.filter(['Close'])

dataset_netflix = netflix_close_data.values
training_netflix = int(np.ceil(len(dataset_netflix) * .9))

train_data_netflix = dataset_netflix[0:training_netflix, :]
test_data_netflix = dataset_netflix[training_netflix:, :]

print(f"Shape of Netflix close data: {netflix_close_data.shape}")
print(f"Length of Netflix training data: {len(train_data_netflix)}")
print(f"Length of Netflix testing data: {len(test_data_netflix)}")

"""
## Scale Netflix Data

Initialize `MinMaxScaler`, then fit and transform the `train_data_netflix` and finally transform `test_data_netflix` using the same scaler.
"""

from sklearn.preprocessing import MinMaxScaler

scaler_netflix = MinMaxScaler(feature_range=(0, 1))

scaled_train_data_netflix = scaler_netflix.fit_transform(train_data_netflix)
scaled_test_data_netflix = scaler_netflix.transform(test_data_netflix)

print(f"Shape of scaled_train_data_netflix: {scaled_train_data_netflix.shape}")
print(f"Shape of scaled_test_data_netflix: {scaled_test_data_netflix.shape}")

"""
## Create Netflix Training/Testing Sequences

Generate sequential `x_train_netflix`, `y_train_netflix`, `x_test_netflix`, and `y_test_netflix` datasets using a 60-day sliding window approach, reshaping them for LSTM input.

"""

x_train_netflix = []
y_train_netflix = []

for i in range(60, len(scaled_train_data_netflix)): # Start from 60th element
    x_train_netflix.append(scaled_train_data_netflix[i-60:i, 0])
    y_train_netflix.append(scaled_train_data_netflix[i, 0])

x_train_netflix, y_train_netflix = np.array(x_train_netflix), np.array(y_train_netflix)
x_train_netflix = np.reshape(x_train_netflix, (x_train_netflix.shape[0], x_train_netflix.shape[1], 1))

x_test_netflix = []
# y_test_netflix is the actual unscaled close prices for the test period
# The test_data_netflix already contains the correct portion of the unscaled dataset
y_test_netflix = test_data_netflix

# Create sequences for x_test_netflix
# The test data needs to include the 'look_back' period from the end of the training data
# or be prepared relative to its own start after the look-back. Given the structure
# scaled_test_data_netflix is directly the scaled test set, we need to adjust for the window.
# However, the conventional way in these examples is to directly use scaled_test_data_netflix
# which implies we don't need elements from the training set. Let's follow the previous pattern.

# Need to create x_test based on scaled_test_data_netflix, but with look-back logic.
# To correctly create x_test, we need the last 60 days from the training set + the test set.
# This is a common point of confusion. Let's re-align with the previous methodology.

# The 'test_data' preparation in previous cells used: `test_data = scaled_data[training - 60:, :]`
# This means `test_data` already includes the necessary look-back period from the training set.
# So we apply the same logic here to `scaled_test_data_netflix` but need to account for its start.

# Let's reconstruct the test_data including the look-back period for Netflix.
# test_data_full_sequence will contain data points from the end of training data up to the end of test data

test_data_full_sequence_netflix = scaled_train_data_netflix[len(scaled_train_data_netflix) - 60:].tolist() + scaled_test_data_netflix.tolist()
test_data_full_sequence_netflix = np.array(test_data_full_sequence_netflix)

for i in range(60, len(test_data_full_sequence_netflix)): # Start from 60th element
    x_test_netflix.append(test_data_full_sequence_netflix[i-60:i, 0])

x_test_netflix = np.array(x_test_netflix)
x_test_netflix = np.reshape(x_test_netflix, (x_test_netflix.shape[0], x_test_netflix.shape[1], 1))

print(f"Shape of x_train_netflix: {x_train_netflix.shape}")
print(f"Shape of y_train_netflix: {y_train_netflix.shape}")
print(f"Shape of x_test_netflix: {x_test_netflix.shape}")
print(f"Shape of y_test_netflix: {y_test_netflix.shape}")

"""
## Build LSTM Model for Netflix

Define the LSTM model architecture for Netflix stock prediction as specified, including two LSTM layers, a Dense layer, a Dropout layer, and an output Dense layer.

"""

model_netflix = keras.models.Sequential()
model_netflix.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train_netflix.shape[1], 1)))
model_netflix.add(keras.layers.LSTM(units=64))
model_netflix.add(keras.layers.Dense(32))
model_netflix.add(keras.layers.Dropout(0.5))
model_netflix.add(keras.layers.Dense(1))

print("Netflix LSTM model architecture defined.")

"""
## Train Netflix Prediction Model

Compile the LSTM model with the Adam optimizer and mean squared error loss, then train it using the prepared `x_train_netflix` and `y_train_netflix` data.

"""

model_netflix.compile(optimizer='adam', loss='mean_squared_error')
print("Netflix LSTM model compiled.")

print("Training Netflix LSTM model...")
history_netflix = model_netflix.fit(x_train_netflix, y_train_netflix, epochs=10, batch_size=32)
print("Netflix LSTM model training complete.")

"""
Now that the Netflix LSTM model has been trained, evaluate its performance on the test data using RMSE and MSE, to assess its predictive accuracy for Netflix stock prices.

"""

predictions_netflix = model_netflix.predict(x_test_netflix)
predictions_netflix = scaler_netflix.inverse_transform(predictions_netflix)

mse_netflix = np.mean(((predictions_netflix - y_test_netflix) ** 2))
rmse_netflix = np.sqrt(mse_netflix)

print(f"Netflix Model Test MSE: {mse_netflix}")
print(f"Netflix Model Test RMSE: {rmse_netflix}")

"""
The next step is to prepare the train and test dataframes for Netflix by adding the predictions to the test set, which will be used for plotting.

"""

train_finetuned = netflix_stock_data[:training_netflix]
test_finetuned = netflix_stock_data[training_netflix:]
test_finetuned['Predictions'] = predictions_netflix

"""
To visualize the model's performance, plot the actual training closing prices, actual testing closing prices, and the predicted testing closing prices for Netflix stock.

"""

plt.figure(figsize=(10, 8))
plt.plot(train_finetuned['Date'], train_finetuned['Close'])
plt.plot(test_finetuned['Date'], test_finetuned[['Close', 'Predictions']])
plt.title('Netflix Stock Close Price - Original Model')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()

"""
## Fine-tune the Model

Train the compiled pre-trained model on the Netflix-specific training data (`x_train_netflix`, `y_train_netflix`).

"""

x_train_general = x_train_netflix
y_train_general = y_train_netflix

print(f"Shape of x_train_general (from Netflix data): {x_train_general.shape}")
print(f"Shape of y_train_general (from Netflix data): {y_train_general.shape}")

"""
Re-define the base LSTM model architecture, compile it, and then train this model on the `x_train_general` and `y_train_general`. After training, save this model to a file, making it available as the 'pre-trained' model for the subsequent fine-tuning step.

"""

pre_trained_model_netflix = keras.models.Sequential()
pre_trained_model_netflix.add(keras.layers.LSTM(units=64,
                                    return_sequences=True,
                                    input_shape=(x_train_general.shape[1], 1)))
pre_trained_model_netflix.add(keras.layers.LSTM(units=64))
pre_trained_model_netflix.add(keras.layers.Dense(32))
pre_trained_model_netflix.add(keras.layers.Dropout(0.5))
pre_trained_model_netflix.add(keras.layers.Dense(1))

pre_trained_model_netflix.compile(optimizer='adam', loss='mean_squared_error')

print("Pre-training the base model on Netflix data...")
history_pretrain_netflix = pre_trained_model_netflix.fit(x_train_general, y_train_general, epochs=5, batch_size=256)

print("Pre-training complete.")

pre_trained_model_netflix.save("general_model_netflix_pretrain.keras")
print("Pre-trained Netflix model saved.")

"""
Now that the base model has been trained and saved, the next step is to load this pre-trained model for fine-tuning. This will allow us to utilize the learned general patterns from the pre-training phase.


"""

fine_tune_model_netflix = tf.keras.models.load_model('general_model_netflix_pretrain.keras')
print("Pre-trained Netflix model loaded successfully for fine-tuning.")

"""
After loading the pre-trained model, it needs to be compiled with an optimizer and loss function, typically with a lower learning rate for fine-tuning, before it can be used for further training.


"""

fine_tune_model_netflix.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
print("Fine-tune model for Netflix data compiled with Adam optimizer and a lower learning rate.")

"""
To fine-tune the model on the Netflix-specific data, I will use the `fit` method of the `fine_tune_model_netflix` with the prepared training data, epochs, and batch size.

"""

print("Fine-tuning the model on Netflix data...")
history_finetune_netflix = fine_tune_model_netflix.fit(x_train_netflix, y_train_netflix, epochs=50, batch_size=32)
print("Fine-tuning on Netflix data complete.")

"""
## Visualize Predictions (Historical, Test, and Future)

### Predict Future Stock Prices
"""

from datetime import timedelta

# Re-load netflix_stock_data to ensure it's in scope
netflix_stock_data = pd.read_csv('NFLX.csv', delimiter=',', on_bad_lines='skip')
netflix_stock_data['Date'] = pd.to_datetime(netflix_stock_data['Date'])

# Get the last 60 days of data from the `netflix_stock_data` for prediction input
last_60_days = netflix_stock_data['Close'].values[-60:].reshape(-1, 1)

# Scale the last 60 days of data using the scaler_netflix (defined in a previous cell)
scaled_last_60_days = scaler_netflix.transform(last_60_days)

# Create an empty list to store future predictions
future_predictions = []

# Number of future days to predict
num_future_days = 30

# Use the scaled_last_60_days as the initial input sequence
current_input = scaled_last_60_days

for _ in range(num_future_days):
    # Reshape current_input to (1, 60, 1) for the model
    current_input_reshaped = current_input.reshape(1, 60, 1)

    # Predict the next day's price using fine_tune_model_netflix
    next_prediction_scaled = fine_tune_model_netflix.predict(current_input_reshaped)[0]

    # Append the prediction to the list
    future_predictions.append(next_prediction_scaled)

    # Update the input sequence: remove the first element and add the new prediction
    current_input = np.append(current_input[1:], [next_prediction_scaled], axis=0)

# Inverse transform the future predictions to get actual prices
future_predictions = scaler_netflix.inverse_transform(future_predictions)

# Generate future dates
last_date = netflix_stock_data['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, num_future_days + 1)]

print("Future predictions generated successfully.")


plt.figure(figsize=(14, 7))

# Plot historical training data
plt.plot(netflix_stock_data['Date'][:training_netflix], netflix_stock_data['Close'][:training_netflix], label='Training Data')

# Plot actual test data
plt.plot(test_finetuned['Date'], test_finetuned['Close'], label='Actual Test Data', color='orange')

# Plot predicted test data
plt.plot(test_finetuned['Date'], test_finetuned['Predictions'], label='Predicted Test Data', color='green', linestyle='--')

# Plot future predictions
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red', linestyle=':')

plt.title('Netflix Stock Price Prediction (Historical, Test, and Future)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))

# Plot historical training data
plt.plot(netflix_stock_data['Date'][:training_netflix], netflix_stock_data['Close'][:training_netflix], label='Training Data')

# Plot actual test data
plt.plot(test_finetuned['Date'], test_finetuned['Close'], label='Actual Test Data', color='orange')

# Plot predicted test data
plt.plot(test_finetuned['Date'], test_finetuned['Predictions'], label='Predicted Test Data', color='green', linestyle='--')

# Plot future predictions
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red', linestyle=':')

plt.title('Netflix Stock Price Prediction (Historical, Test, and Future)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()


fine_tune_model = tf.keras.models.load_model('general_model_epoch5_batchsize256.keras')

print("Pre-trained model loaded successfully for Netflix data.")

"""
To train the model on the Netflix-specific data, I will use the `fit` method of the `fine_tune_model` with the provided training data, epochs, and batch size.

"""


# Reload the pre-trained model
fine_tune_model = tf.keras.models.load_model('general_model_epoch5_batchsize256.keras')
print("Pre-trained model loaded successfully for fine-tuning on Netflix data.")

# Compile the loaded model with a lower learning rate for fine-tuning
fine_tune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
print("Fine-tune model compiled with Adam optimizer and a lower learning rate for Netflix data.")

print("Fine-tuning the model on Netflix data...")
history_finetune_netflix = fine_tune_model.fit(x_train_netflix, y_train_netflix, epochs=50, batch_size=32)
print("Fine-tuning on Netflix data complete.")


# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history_finetune_netflix.history['loss'], label='Training Loss')
plt.title('Fine-tuned Model Training Loss for Netflix Stock')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()
