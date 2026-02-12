# Yes-Bank-Stock-Prediction
This project predicts the monthly closing price of Yes Bank stock, focusing on the impact of the 2018 management crisis. The goal was to develop a model robust enough to handle high volatility and non-linear trends.

## Models Implemented
Linear Regression: Baseline performance.

Lasso (L1): For feature selection and noise reduction.

ElasticNet: Final model using a combination of L1 and L2 regularization to handle multi-collinearity.

## Key Metrics
R2 Score: > 95%

Performance: The model accurately tracks price movements even during the post-crisis recovery phase.

## Features Used
Log-transformed values of Open, High, Low, and Previous Month's Close, along with engineered features like Price Spread and OHLC Average.

