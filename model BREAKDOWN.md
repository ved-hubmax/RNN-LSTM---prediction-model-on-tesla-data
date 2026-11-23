# Tesla RNN & LSTM Model Breakdown

## What this project does

This notebook is a time-series forecasting experiment using historical Tesla prices. Its target is the **next trading day's `Open` price**, estimated from the preceding 50 trading days of `Open` prices.

It compares two neural-network approaches:

- A stacked **SimpleRNN** model.
- A stacked **LSTM** model.

The notebook is educational and explicitly says it does not provide trading tips. Its forecasts are not buy or sell instructions.

## Data and tools used

The notebook uses:

- **Tesla CSV data:** 1,692 daily records, dated June 29, 2010 to March 17, 2017 in the saved outputs.
- **pandas and NumPy:** load, select, and reshape the data.
- **scikit-learn `MinMaxScaler`:** scale the model input and target to the range 0–1, and convert predictions back to dollars.
- **TensorFlow/Keras:** define and train the recurrent neural networks.
- **Matplotlib, Seaborn, and Plotly:** inspect and visualize the series and predictions.

The CSV contains `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, and `Adj Close`. The notebook checks the data types, summary statistics, and missing values; the saved output reports no missing values. It converts `Date` to a datetime type.

**Important:** although the CSV has several market columns, the models use only **`Open`**. They do not use `High`, `Low`, `Close`, `Volume`, or `Adj Close` as model inputs. The target is also `Open`.

The notebook loads the CSV from a path relative to the project root:

```text
data/Tesla.csv.csv
```

Keep the CSV in the repository's `data/` folder when running the notebook.

## Features and model design

### Input window and target

The notebook makes overlapping sliding windows of 50 observations:

```text
previous 50 daily Open prices  ->  next daily Open price
```

Fifty trading days is roughly ten trading weeks. The first 70% of the records (1,184 rows) are used for training; the remaining 508 rows form a later, chronological validation period. The saved output gives the training period as June 29, 2010–March 12, 2015 and validation as March 13, 2015–March 17, 2017.

The split is chronological rather than randomly shuffled, which is appropriate for a time series. The scaler is fitted on training prices and then reused for validation inputs. Predictions are inverse-transformed to dollar prices before plotting.

The validation windows use the latest **actual** prices available before each validation day. This evaluates repeated one-day-ahead forecasts with updated history; it is not a simulation of forecasting many future days in a row using the model's own previous predictions.

### SimpleRNN model

The first model has:

- Four stacked `SimpleRNN` layers, each with 50 units.
- `tanh` activation.
- Dropout layers with a 0.2 rate after the second, third, and fourth recurrent layers.
- A one-unit `Dense` output for the predicted scaled price.
- Adam optimizer and mean squared error (MSE) training loss.
- Up to 50 epochs, batch size 32.
- An `EarlyStopping` callback that monitors **training loss**, waits five epochs without improvement, and restores the best training-loss weights.

### LSTM model

The second model has:

- Two stacked LSTM layers, each with 64 units.
- A 32-unit `Dense` hidden layer.
- A one-unit `Dense` output for the predicted scaled price.
- Adam optimizer and MSE training loss.
- 50 epochs and batch size 32.

The LSTM early-stopping callback is present only as commented-out code, so it is **not used** during LSTM training.

LSTM cells have gates and a cell state that can help preserve useful information over longer sequences than a plain SimpleRNN. That architectural advantage does not guarantee better forecasts. These two models also differ in layer count, units, dense layers, and regularization, so their results are not a controlled, layer-for-layer comparison.

## How to read the graphs

- **Open-price histogram:** shows how often prices fell into each price range. It describes the distribution, not when prices occurred or what will happen next.
- **OHLC line charts:** show the historical Open, High, Low, and Close series over dates. The interactive Plotly range slider and unified hover make it easier to inspect shorter periods.
- **Before/after scaling charts:** show the same training Open-price sequence in dollars and on a 0–1 scale. Scaling changes the units, not the underlying price movements, and is not evidence of model accuracy.
- **Training actual-versus-predicted chart:** shows how closely the model fits the data it learned from. A close overlap here does not establish that it can forecast unseen data.
- **Validation actual-versus-predicted charts:** compare the forecast and actual Open prices for the later, held-out period. Close tracking is encouraging, while lag around sharp reversals or large errors suggests the forecast is not capturing those changes. A plot is qualitative; it cannot replace numerical validation metrics.
- **Full timeline chart:** places training and validation history on a date axis, marks the split, and overlays the model's training fit and validation forecasts.

## Performance and accuracy: what is and is not measured

**The notebook does not report a numerical validation accuracy.** It imports `mean_absolute_error`, `mean_squared_error`, and `r2_score`, but the notebook does not call them. It also does not calculate classification accuracy, trading returns, or a profitability metric.

“Accuracy” in the usual classification sense is not the right measure for this task: the models predict a continuous price, not a class such as up/down. For a useful assessment, compute at least:

- **MAE (Mean Absolute Error):** average absolute prediction error in dollars; lower is better.
- **RMSE (Root Mean Squared Error):** error in dollars that penalizes large misses more heavily; lower is better.
- **R²:** how much variation the predictions explain relative to a simple mean baseline; it can be negative when predictions are worse than that baseline.
- **Directional accuracy (optional):** how often the forecast correctly predicts the next price movement's direction. This is not the same as price accuracy or trading profitability.

Calculate price-error metrics using the **inverse-transformed validation predictions** and actual validation Open prices, not the scaled training logs. Report RNN and LSTM results separately, and compare them with a simple baseline such as predicting that the next Open will equal the most recently observed Open. A model should demonstrate improvement over that baseline before its forecasts are treated as useful.

The training cells display training loss and training MAE while fitting. These are not validation scores and cannot establish generalization. In particular:

- The RNN early-stopping callback watches training loss, not validation loss.
- The LSTM trains for all 50 epochs; its early-stopping callback is commented out.
- Neither model's `fit()` call supplies validation data or reports validation loss.
- The notebook plots held-out validation predictions, but does not calculate error metrics from them.

Consequently, the notebook supports a visual demonstration of forecasting, but **does not provide enough evidence to claim a specific accuracy level or that one model is better**.

## How validation works in this notebook

1. Records are split in time order: the earlier 70% is training data and the later 30% is held out.
2. The scaler learns its minimum and maximum from training prices only.
3. Each training example uses 50 earlier training Open prices to predict the next training Open.
4. Each validation example uses 50 earlier actual Open prices, including earlier observations from the validation period once they have occurred.
5. Each model predicts the validation period, and its scaled output is converted back to dollars for plotting against actual Open prices.

This setup tests one-step-ahead predictions over a later historical period. It does not show how the model would perform after retraining on recent data, in a different market regime, or in a real order-execution environment. A stronger evaluation would use walk-forward testing across multiple time periods, keep a final untouched test period, and compare against simple baselines.

## Using forecasts in a responsible trading-strategy evaluation

The current notebook does **not** create a trading strategy: it has no entry or exit rules, position sizing, stop rules, transaction-cost model, portfolio accounting, or backtest. A predicted Open price alone is not a trading signal and does not indicate whether a trade's expected return is positive after risk and costs.

If you extend the project for research, treat it as a hypothesis to test:

1. Define the decision time and use only information actually available then. The notebook predicts an Open from prior Opens; do not use that day's actual Open to make a decision supposedly made before that Open.
2. Define explicit entry, exit, and position-sizing rules before examining results. Include a threshold that accounts for fees, bid–ask spread, slippage, and forecast uncertainty.
3. Backtest chronologically or with walk-forward retraining. At every simulated decision, train only on earlier data and avoid using future information in preprocessing, tuning, or signal generation.
4. Compare with simple alternatives (including a no-trade policy and a naive price forecast) and a relevant benchmark.
5. Include realistic trading costs and constraints; track returns, drawdown, risk-adjusted performance, and turnover, not only price-prediction error.
6. Paper-trade and monitor the approach out of sample before considering any real-money use. Historical performance does not guarantee future results.

This is an educational explanation, not financial advice or a recommendation to trade Tesla or any other security.
