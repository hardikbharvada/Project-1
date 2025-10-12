-----

# Ensemble Deep Learning for Probabilistic Carbon Price Forecasting

[cite\_start]This repository contains the official implementation for the paper: **Ensemble Deep Learning for Probabilistic Forecasting of EU ETS Carbon Prices**[cite: 3].

[cite\_start]This project develops a probabilistic deep learning framework to forecast carbon prices in the European Union Emissions Trading System (EU ETS), moving beyond simple point estimates to capture the high volatility and uncertainty inherent in carbon markets[cite: 14].

## 📖 Abstract

[cite\_start]Forecasting carbon prices in the European Union Emissions Trading System (EU ETS) requires methods that can capture high volatility and structural uncertainty[cite: 14]. [cite\_start]This study develops a deep learning framework for probabilistic forecasting, comparing distribution-free approaches (quantile and tube regression) with parametric models based on Mixture Density Networks (MDNs)[cite: 15]. [cite\_start]Our results show that MDN variants produce sharper and better-calibrated prediction intervals in both univariate and multivariate settings[cite: 16]. [cite\_start]To address variability and parameter sensitivity, we introduce ensemble MDNs optimized with the Optuna Bayesian framework, leading to more stable forecasts[cite: 17]. [cite\_start]The inclusion of exogenous variables—such as crude oil, natural gas, $CO_{2}$ emissions, and the Euronext 100 index—further enhances model accuracy[cite: 18]. [cite\_start]The proposed TCN+MDN ensemble with exogenous predictors achieves the narrowest prediction intervals and best performance, establishing a robust benchmark for probabilistic carbon price forecasting[cite: 19].

## ✨ Key Features

  * [cite\_start]**Probabilistic Forecasting**: Shifts from traditional point forecasting to generating prediction intervals that quantify uncertainty, which is critical for risk management in volatile carbon markets[cite: 33, 34].
  * [cite\_start]**Advanced Deep Learning Models**: Implements and compares a wide range of deep auto-regressive architectures, including RNN, LSTM, GRU, and Temporal Convolutional Networks (TCNs)[cite: 66].
  * [cite\_start]**Mixture Density Networks (MDNs)**: Utilizes MDNs to estimate the full predictive density, leading to sharper and better-calibrated intervals compared to distribution-free methods[cite: 16, 76].
  * [cite\_start]**Hyperparameter Optimization**: Employs the Bayesian optimization framework `Optuna` to efficiently tune model hyperparameters, balancing prediction interval width and coverage probability[cite: 17, 386, 387].
  * [cite\_start]**Ensemble Methodology**: Incorporates an ensemble approach to account for both model uncertainty and data uncertainty, resulting in more stable and robust forecasts[cite: 17, 81, 394].
  * [cite\_start]**Exogenous Variables**: Demonstrates that integrating external factors like energy prices, economic indices, and emissions data significantly improves forecast sharpness and distributional accuracy[cite: 18, 84, 85].
  * [cite\_start]**Comprehensive Comparison**: Provides a comparative analysis of 34 different deep probabilistic forecasting models to establish a state-of-the-art benchmark[cite: 89].

## ⚙️ Methodology

1.  **Data Collection & Preparation**:

      * [cite\_start]Historical European Union Allowance (EUA) carbon prices (OHLC, volume) were gathered from Investing.com for the period of Jan 4, 2021, to Aug 2, 2024[cite: 335].
      * [cite\_start]Exogenous variables including crude oil prices, natural gas prices, daily $CO_{2}$ emissions, and the Euronext 100 index were collected from various sources to enrich the input features[cite: 336].
      * [cite\_start]The data is processed using a sliding window approach, and all features are normalized[cite: 279, 374].

2.  **MDN-based Deep Probabilistic Forecasting**:

      * [cite\_start]Deep auto-regressive models (GRU, TCN, LSTM) are trained to predict the parameters of a Gaussian mixture model[cite: 376, 377].
      * [cite\_start]The models minimize the Negative Log-Likelihood (NLL) to estimate the conditional probability density $f(X_{t+1}|R_{t})$[cite: 381, 382].

3.  **Hyperparameter Optimization with Optuna**:

      * [cite\_start]The `Optuna` framework is used for efficient Bayesian hyperparameter optimization[cite: 387, 388].
      * [cite\_start]A custom score function is defined to find optimal parameters that achieve the target coverage (95%) while minimizing the Mean Prediction Interval Width (MPIW)[cite: 389].

4.  **Ensembling for Uncertainty Quantification**:

      * [cite\_start]To account for model uncertainty arising from stochastic elements in training, an ensemble approach is used[cite: 392, 394].
      * [cite\_start]The model is trained `m` times (in our case, `m=5`) with the best hyperparameters, and the resulting prediction intervals are aggregated to produce a final, more robust interval that captures total predictive uncertainty[cite: 400, 410].

5.  **Evaluation**:

      * The quality of the probabilistic forecasts is assessed using standard metrics:
          * [cite\_start]**Prediction Interval Coverage Probability (PICP)**: Measures the percentage of true values that fall within the predicted interval[cite: 253].
          * **Mean Prediction Interval Width (MPIW)**: Measures the average sharpness or width of the prediction intervals. [cite\_start]Narrower is better, given adequate coverage[cite: 259].
          * [cite\_start]**Continuous Ranked Probability Score (CRPS)**: Evaluates both the calibration and sharpness of the entire predictive distribution[cite: 269].

## 📈 Results

Our experiments demonstrate a clear progression in performance from simple point forecasts to the final proposed ensemble model.

  * [cite\_start]**Point forecasts are insufficient**: While GRU-based models performed best, the marginal improvements are minimal and fail to capture the risk present in volatile carbon markets[cite: 425, 426].
  * [cite\_start]**MDNs outperform distribution-free methods**: MDN-based models consistently achieved a better trade-off between coverage (PICP) and sharpness (MPIW) compared to Quantile and Tube regression models[cite: 479].
  * **Multivariate and Exogenous data are key**:
      * [cite\_start]Adding OHLC data (multivariate) reduced MPIW compared to using only the closing price[cite: 526].
      * [cite\_start]Incorporating exogenous variables led to the sharpest intervals and best-calibrated distributions[cite: 586].
  * [cite\_start]**TCN+MDN is the state-of-the-art**: The proposed `TCN+MDN` model, enhanced with an ensemble strategy and exogenous variables, emerged as the top-performing model[cite: 587, 594]. It achieved the:
      * [cite\_start]Lowest Mean Prediction Interval Width (MPIW) of **6.75**[cite: 594].
      * [cite\_start]Lowest Continuous Ranked Probability Score (CRPS) of **0.93**[cite: 594].
      * [cite\_start]Highest overall efficiency and robustness[cite: 588].
    

## 💻 Setup and Usage

### Prerequisites

  * Python 3.8+
  * PyTorch
  * Optuna
  * Pandas
  * NumPy
  * Scikit-learn

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/probabilistic-carbon-forecasting.git
    cd probabilistic-carbon-forecasting
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Models

1.  **Data Preparation**:
    Place your time-series data in the `data/` directory. [cite\_start]Ensure it is pre-processed as described in the paper[cite: 334].

2.  **Training a Model**:
    To train a model, run the main script with the desired configuration. For example, to train the TCN+MDN model with exogenous variables and Optuna optimization:

    ```bash
    python train.py --model tcn_mdn --use_exogenous --optimize_hparams --n_trials 100
    ```

3.  **Evaluation**:
    To evaluate a trained model on the test set:

    ```bash
    python evaluate.py --model_path checkpoints/tcn_mdn_exogenous_best.pt
    ```

}
```
