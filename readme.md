# CoinCast: Precision Bitcoin Market Forecasting

## CMPE 255: Section 49 Project 2024

**Team Synergy**

- Shubham Kothiya
- Soumya Bharathi
- Rutuja Patil
- Mann Nada 

---

## Abstract

The volatility of Bitcoin prices creates significant challenges for investors and financial analysts. This project addresses the problem by developing robust time series forecasting models to predict Bitcoin prices using historical Open, High, Low, Close (OHLC), and Volume data. Models explored include ARIMA, Random Forest, Gradient Boosting, and LSTM. Ablation studies and hyperparameter tuning were performed to optimize model performance. The CRISP-DM methodology was adopted to ensure a systematic approach, including data understanding, preparation, modeling, and evaluation. The resulting models were evaluated on metrics such as MSE, RMSE, and R², with visualizations to demonstrate performance. A Gradio-based application was deployed for inference, with production pipelines implemented on a cloud platform.

---

## Project Overview

### 1. Introduction 🔍

Bitcoin price forecasting is a crucial task due to the unpredictable nature of the cryptocurrency market. Accurate predictions can assist investors in making informed financial decisions and understanding market dynamics. This project uses a comprehensive dataset and machine learning techniques to forecast Bitcoin prices. Models like ARIMA, Random Forest, Gradient Boosting, and LSTM were applied, tuned, and evaluated systematically. Key deliverables include:

- A runnable Colab notebook.
- A Gradio demo app for real-time predictions.
- A cloud-based production pipeline.

---

### 2. Related Work

- **Traditional Time Series Models**: ARIMA is commonly used for linear time series predictions.
- **Machine Learning Techniques**: Random Forest and Gradient Boosting models handle non-linear trends effectively.
- **Deep Learning**: LSTMs excel at learning from sequential data and are widely used for time series forecasting.

Our approach combines these methods, compares their performance, and explores the impact of hyperparameter tuning and data preprocessing.

---

### 3. Data 📊

- **Data Source**: Bitcoin historical OHLC and Volume data sourced from CoinDesk and Kaggle datasets.
- **Data Size**: ~6.7 million records spanning multiple years.

#### Preprocessing:

- Handled missing values.
- Aggregated 1-minute data to daily mean prices.
- Normalized data using `MinMaxScaler` for LSTM models.
- Split data into 75% training and 25% testing.

---

### 4. Methods 🛠️

#### CRISP-DM Methodology

The CRISP-DM framework ensured a structured workflow for the project:

1. **Business Understanding**: Predict Bitcoin prices to aid investors and analysts.
2. **Data Understanding**: Explored trends, outliers, and correlations using EDA.
3. **Data Preparation**: Handled missing data, scaling, and data splits.
4. **Modeling**: Applied and tuned multiple models:
   - **ARIMA**: For linear predictions.
   - **Random Forest & Gradient Boosting**: For non-linear trends.
   - **LSTM**: For deep learning-based sequential forecasting.
5. **Evaluation**: Models were evaluated using metrics such as MSE, RMSE, MAE, and R².
6. **Deployment**: Gradio app for real-time predictions and cloud-based production pipeline.

#### Ablation Study and Tuning:

- Explored hyperparameters, such as ARIMA's (p, d, q) values and LSTM layer sizes.
- Compared performance with and without normalization and dropout layers.

---

### 5. Experiments and Results 🌟

The following experiments were performed, and the models' results are summarized below:

| **Model**             | **MSE**        | **RMSE**  | **R²** |
| --------------------- | -------------- | --------- | ------ |
| **ARIMA**             | 347,173,971.42 | 18,630.65 | 0.72   |
| **Random Forest**     | 37,444,453.09  | 6,116.64  | 0.85   |
| **Gradient Boosting** | 52,355,774.68  | 7,234.95  | 0.81   |
| **LSTM**              | 12,136,865.00  | 3,485.26  | 0.91   |

#### Visualization Techniques:

- Plots of actual vs. predicted prices.
- Training loss and validation loss graphs.
- Scatter plots to demonstrate correlations.

**Conclusion**: LSTM achieved the best performance due to its ability to learn sequential dependencies in data.

---

### 6. Deployment 🚀

#### Live Demo: Gradio-CoinCast on Hugging Face Spaces

The trained model is integrated into a Gradio-based web application to perform real-time inference:

- **Interactive Demo**: Users can input parameters and predict future Bitcoin prices instantly.

#### Cloud Deployment Pipeline

The deployment pipeline was implemented using Hugging Face Spaces for hosting, automated via GitHub Actions:

**Deployment Steps**:

1. The trained model is saved in the required format (ONNX/Pickle/TF SavedModel).
2. GitHub Actions automates deployment by pushing updates to the Hugging Face repository.
3. The Gradio application is hosted on Hugging Face Spaces for public access.

**Benefits of Hugging Face Spaces**:

- Simple and scalable hosting platform.
- Integration with Gradio for interactive applications.
- Ensures seamless deployment and accessibility for end-users.

---

### 7. Conclusion

This project successfully applied ARIMA, Random Forest, Gradient Boosting, and LSTM models to predict Bitcoin prices. **LSTM** outperformed other models with an RMSE of **3,485.26**. The deployment of a Gradio application and a production pipeline demonstrates the project’s practical usability.

#### Future Work:

- Incorporating external features like social media sentiment analysis or global economic indicators.

---

## Open-Source Availability 🌐

All project artifacts and trained models are open-source and hosted on the Hugging Face platform for public access:

- **ARIMA Model**: [ARIMA Forecasting Model](#)
- **Random Forest Model**: [Random Forest Forecasting Model](#)
- **Gradient Boosting Model**: [GradientBoost Forecasting Model](#)
- **LSTM Model**: [LSTM Fine-Tuned Model](#)

This open-source approach ensures accessibility, transparency, and reproducibility, encouraging further collaboration and innovation.

---

# Bitcoin Historical Data Analysis

## Phase 2: Data Understanding

In this phase, we explored the historical Bitcoin dataset to gain insights into the data structure and quality.

### Data Source
- The dataset includes columns for **Open**, **High**, **Low**, **Close** prices, **Volume**, and **Timestamps**.

### Key Steps Performed
1. **Timestamp Conversion**:
   - Converted timestamps into a readable date format.
2. **Data Aggregation**:
   - Aggregated data to a daily level by calculating the mean values for OHLC prices and Volume.
3. **Univariate Analysis**:
   - Calculated **average**, **minimum**, and **maximum** prices.
   - Checked for **outliers** using Interquartile Range (IQR).
   - Visualized the data using **line plots** to observe price trends over time.

This understanding provided a foundation for further preprocessing and modeling.

---

## Dependencies Installation
Before proceeding, install the necessary libraries:

```bash
pip install gradio
```

**Libraries Installed:**
- aiofiles
- fastapi
- ffmpy
- gradio-client
- markupsafe
- numpy, pandas, and seaborn (if not already installed)
- python-multipart
- ruff
- safehttpx
- semantic-version
- starlette
- tomlkit
- uvicorn
- pydub

---

## Required Libraries
The following libraries were used for data manipulation, visualization, and model building:

```python
# Import necessary libraries
import math                 # For mathematical operations
import numpy as np          # For numerical operations
import pandas as pd         # For data manipulation and analysis
import seaborn as sns       # For data visualization
sns.set_style('whitegrid')  # Set seaborn style to whitegrid
import matplotlib.pyplot as plt  # For plotting graphs
plt.style.use("fivethirtyeight")  # Use 'fivethirtyeight' style for matplotlib plots

# Keras libraries for building neural network models
from keras.models import Sequential  # For sequential model building
from keras.callbacks import EarlyStopping  # For early stopping during model training
from keras.layers import Dense, LSTM, Dropout  # Layers for neural network

# Scikit-learn libraries for preprocessing and evaluation
from sklearn.preprocessing import MinMaxScaler  # For data normalization
from sklearn.model_selection import train_test_split  # For train-test split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Model evaluation

import warnings  # To ignore warnings
warnings.simplefilter('ignore')
```

---

## Dataset Download
The Bitcoin dataset was downloaded from Kaggle using `kagglehub`. Make sure to set up your Kaggle API key.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
print("Path to dataset files:", path)
```

---

## Dataset Loading
We loaded the Bitcoin historical dataset and displayed its basic structure.

```python
import os

# Path to dataset
csv_file = "btcusd_1-min_data.csv"
full_path = os.path.join(path, csv_file)

# Load dataset
df = pd.read_csv(full_path)

# Display first and last rows
print("First 5 rows of the dataset:")
print(df.head())
print("Dataset shape:", df.shape)
print("List of columns/features:", df.columns.tolist())
print("Detailed Info:")
print(df.info())

# View dataset's tail
print(df.tail())
```

---

## Data Exploration Results
After exploring the dataset, the following observations were made:
- The dataset contains timestamps and price data at a 1-minute resolution.
- Missing values were identified, and handling strategies (e.g., interpolation) will be applied.
- Outliers in price data were detected using IQR.
- Visualizations revealed trends and fluctuations in Bitcoin price data.

The insights gained here serve as the groundwork for data preprocessing, feature engineering, and model building.

---

## Next Steps
- Handle missing values.
- Normalize the dataset for LSTM input.
- Build and train neural network models for Bitcoin price prediction.
- Evaluate model performance using RMSE, MAE, and R-squared metrics.

---

**Note:** Ensure that your environment is set up with all the required libraries and API keys for seamless execution of the above code.

# Phase 3: Data Preparation

In this phase, we prepared the dataset for modeling by cleaning, transforming, and normalizing the data to ensure it is suitable for building forecasting models like LSTM.

---

## Key Steps Performed

### 1. Timestamp Conversion and Grouping by Date
- Converted the `Timestamp` column into a human-readable `Date` format.
- Grouped data by date to compute daily averages for key features: **`Open`**, **`Close`**, **`High`**, **`Low`** prices, and **`Volume`**.

---

### 2. Handling Missing Values
- Identified and handled missing values to ensure dataset consistency and completeness.

---

### 3. Exploratory Data Analysis (EDA)

#### Bitcoin Price Insights
- Calculated key statistics:
  - **Average Opening Price**
  - **Highest Opening Price** and its corresponding date
  - **Lowest Opening Price**
- Analyzed overall price trends over time.

#### Outlier Detection
- Detected outliers in critical columns such as **`Open`**, **`Close`**, and **`Volume`** using statistical methods to identify anomalies.

#### Visualizations
- **Price Trends**: Line plots for `Open` and `Close` prices to observe trends and volatility.
- **Volume Analysis**: Plots showing trading volume to identify spikes or unusual activity.
- **Correlation Analysis**: Examined relationships between key features:
  - Correlation between `Open` and `Close` prices.
  - Correlation between price changes and trading volume.

---

### 4. Normalizing the Data
- Normalized the data to a range between `0` and `1` to ensure compatibility with LSTM models.
- Normalization helps avoid features with larger scales dominating the learning process.

---

### 5. Splitting the Dataset
- Split the dataset into:
  - **Training Set**: 75%
  - **Testing Set**: 25%
- Ensured the model is trained on one subset and evaluated on unseen data.

---

### 6. Creating Sequences for LSTM
- Used a **sliding window approach** with `60-time steps` to prepare input-output pairs for LSTM:
  - Past `60-time steps` are used to predict the next time step.

---

## Key Insights

1. **Trends in Prices**
   - Bitcoin prices showed significant volatility and long-term growth trends.

2. **Outliers**
   - Identified major market events through outliers in price data.

3. **Correlation Analysis**
   - Strong positive correlation between `Open` and `Close` prices.
   - Negative correlation between trading volume and price changes.

4. **Data Normalization**
   - Successfully scaled data to ensure compatibility with LSTM models.

---

## Outputs from Phase 3
- Cleaned and preprocessed dataset ready for modeling.
- Training and testing datasets created.
- Input sequences structured for LSTM using a `60-time step` window.

---
#  Phase 4: Modeling

## Models Implemented

### 1. **ARIMA**
- Suitable for linear trends and seasonality.
- Tuned parameters for optimal performance.

### 2. **Random Forest Regressor**
- Captures non-linear relationships.
- Efficient for large datasets.

### 3. **Gradient Boosting (XGBoost)**
- Boosts weak learners to improve performance.

### 4. **LSTM (Long Short-Term Memory)**
- Deep learning model for sequential data.

## Model Evaluation

### ARIMA:
- **RMSE:** 20519.2
- **R-squared (R2):** 0.05

### Random Forest:
- **RMSE:** 6759.12
- **R-squared (R2):** 0.88

### Gradient Boosting:
- **RMSE:** 7872.76
- **R-squared (R2):** 0.84

# Phase 5 Evaluation

## Models Used

1. **ARIMA (AutoRegressive Integrated Moving Average)**: A time-series forecasting method suitable for predicting future values based on past observations.
2. **Random Forest**: A machine learning algorithm using an ensemble of decision trees to perform regression tasks.
3. **Gradient Boosting**: A machine learning technique that builds a series of decision trees, each correcting the errors of the previous one.

## Evaluation Metrics

The following metrics are used to evaluate model performance:
- **MSE (Mean Squared Error)**: Measures the average squared difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error)**: Provides the error in the same units as the target variable.
- **MAE (Mean Absolute Error)**: Measures the average magnitude of prediction errors.
- **R² (R-squared)**: Represents the proportion of variance explained by the model.

### Model Performance Comparison:

| Model             | MSE                | RMSE      | R²   |
|-------------------|--------------------|-----------|------|
| ARIMA             | 347,173,971.42     | 18,630.65 | 0.72 |
| Random Forest     | 37,444,453.09      | 6,116.64  | 0.85 |
| Gradient Boosting | 52,355,774.68      | 7,234.95  | 0.81 |
| LSTM              | 12,136,865.00      | 3,485.26  | 0.91 |

## Steps to Run

1. **Install Dependencies**:
   The following Python libraries are required to run this project:
   ```bash
   pip install numpy pandas matplotlib scikit-learn statsmodels

## Phase 6: Deployment on Hugging Face Spaces

The project has been successfully deployed on **Hugging Face Spaces** using the Gradio interface. You can interact with the models directly, visualize the predicted Bitcoin prices for the next 60 days, and gain insights into the performance of different forecasting models.

### Live Demo
You can access the live demo of the project by visiting the following link:

- [Live Demo](https://huggingface.co/spaces/shubh7/gradio-CoinCast)

The demo allows you to input data and observe predictions made by the models including **ARIMA**, **Random Forest**, **Gradient Boosting**, and **LSTM**.

## Phase 7: Project Artifacts Open Source

All models used in this project are open-sourced and hosted on Hugging Face for public access. These models can be downloaded, used, or fine-tuned as per your requirements.

### Available Models:
1. **ARIMA Model**: Time series forecasting model.
   - [ARIMA Model on Hugging Face](https://huggingface.co/shubh7/arima-forecasting-model)

2. **Random Forest Model**: Ensemble machine learning model for regression tasks.
   - [Random Forest Model on Hugging Face](https://huggingface.co/shubh7/RandomForest-forecasting-model)

3. **Gradient Boosting Model**: Another ensemble learning method used for regression.
   - [Gradient Boosting Model on Hugging Face](https://huggingface.co/shubh7/GradientBoost-forecasting-model)

4. **LSTM Model**: Deep learning model for sequential data, fine-tuned for Bitcoin price forecasting.
   - [LSTM Model on Hugging Face](https://huggingface.co/shubh7/LSTM-finetuned-model)

### How to Use These Models:
You can download these models from Hugging Face and integrate them into your own forecasting applications. The models are stored with pre-trained weights, so they are ready to make predictions out of the box. For more detailed usage, visit each model page for code examples and additional instructions.

## License

This project and all model artifacts are open source and licensed under the [MIT License](LICENSE).

---

Feel free to explore the live demo and models. Contributions to improve or extend the project are always welcome!
# Youtube Video Link: https://youtu.be/YdtHgoyENoA 
