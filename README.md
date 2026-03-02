# Stock Predictor Project
This project implements and optimizes a Long Short-Term Memory (LSTM) neural network for time-series stock price prediction. It covers the end-to-end machine learning workflow, from data preprocessing to advanced techniques like Transfer Learning.

## Table of Contents
*    Overview
*    Installation and Setup
*    Model Development and Optimization
     * Baseline Model (AAPL)
     *  Hyperparameter Tuning (AMZN)
      * Transfer Learning Implementation (AMZN)
*    What is Transfer Learning?
*    Data Sources


## Overview
This project explores how deep learning models capture temporal patterns in financial data using:

*    **Time-Series Forecasting**: Predicting 'Close' prices based on a 60-day historical window.
*    **Optimization**: Systematically finding the best architecture using Keras Tuner.
*    **Knowledge Transfer**: Reusing patterns from a general market model to improve specific stock predictions.

## Installation and Setup
Ensure you have Python 3.8+ installed.
1. Clone the repository
```
git clone https://github.com/shjawale/stock-predictor
cd stock-predictor-project
```

2.  Install dependencies
```
pip install pandas numpy matplotlib seaborn tensorflow keras-tuner scikit-learn
```

3. Download the all_stocks_5yr.csv file from Kaggle and place it in the project root directory. It can be found at [stock dataset](https://www.kaggle.com/datasets/rohitjain454/all-stocks-5yr).

## Model Development and Optimization
### Baseline Model (AAPL)
Establishing a performance benchmark using Amazon Inc. (AMZN) data.

*    **Architecture**: Two stacked LSTM layers (64 units) with Dropout (0.5).
*    **Preprocessing**: Data normalized using MinMaxScaler (0,1).
*    **Evaluation**: The baseline model effectively captures major price trends with a low relative error, providing a solid foundation for further optimization.

### Hyperparameter Tuning (AMZN)
Optimizing the model for Amazon (AMZN) stock using RandomSearch.

*    **Search Space**: LSTM units (32–128), Dense units (16–64), and learning rates (10<sup>-4</sup> to 10<sup>-2</sup>).
*    **Observation**: Initial high errors revealed a critical sensitivity to data scaling, necessitating precise stock-specific refitting to accommodate AMZN's higher price point.

### Transfer Learning Implementation (AMZN)
Applying pre-trained "market knowledge" to Amazon (AMZN) data.

*    **Method**: A "Global Model" was pre-trained on a diverse multi-stock dataset.
*    **Process**: The early LSTM layers were frozen, and the final Dense layers were retrained on AMZN's specific price history.
*    **Outcome**: This resulted in faster convergence and better adaptation to specific volatility compared to training from scratch.

## What is Transfer Learning?
Transfer Learning is a technique where a model developed for a general task (the Source Task) is reused as the starting point for a model on a more specific task (the Target Task).
### Core Concepts

*    **Application**:  I trained a complex model on a vast dataset to learn universal market dynamics and technical patterns. This model is then fine-tuned on a single stock like Amazon.
*    **Key Benefits**: 
     *    **Efficiency**: Drastically reduces training time and computational power.
     *    **Data Scarcity**: Improves performance for specific stocks where historical data might be limited.
     *    **Robustness**: Diverse pre-training helps the model generalize better to unseen market conditions.
*    Challenges: 
     *   **Domain Mismatch**: Tech stocks may behave very differently than Utility stocks.
     *   **Non-Stationarity**: Markets evolve; historical patterns may differ significantly from future trends.
      *  **Distribution Shifts**: Major economic events can change the statistical properties of the data, making past patterns less relevant.

## Data Source
I used the [all_stocks_5yr](https://www.kaggle.com/datasets/rohitjain454/all-stocks-5yr) dataset, containing five years of historical daily stock prices for various companies. The dataset is provided by Rohit Jain on Kaggle.

