# PakWheels Car Price Prediction

This project builds a machine learning model to predict used car prices in Pakistan based on data scraped from PakWheels. The model uses a linear regression algorithm with a preprocessing pipeline including One-Hot Encoding for categorical features.

## Dataset

- **Source**: [PakWheels.com](https://www.pakwheels.com)
- **Format**: CSV
- **Path**: `D:/Datasets/pakwheels.csv`
- The dataset includes features like `title`, `model`, `mileage`, `fuel_type`, `transmission`, `registered`, `color`, `assembly`, `engine_capacity`, `vehicle_age`, and `price`.

## Data Preprocessing

- Dropped irrelevant columns: `post_date`, `price_category`, `mileage_category`, `post_day_of_week`.
- Replaced `0` values in the `price` column with the **mean** of non-zero prices.
- Formatted `price` values to 2 decimal places.
- Reduced `title` column to first two words (to capture car make and model).
- Filtered dataset to include only top 24 most frequent car titles (e.g., Toyota Corolla, Honda Civic, etc.).
- Reset indices for clean processing.

## Model Pipeline

- **Input Features**: 
  - Categorical: `title`, `city`, `fuel_type`, `transmission`, `registered`, `color`, `assembly`
  - Numerical: `model`, `mileage`, `engine_capacity`, `vehicle_age`
- **Transformer**: `ColumnTransformer` using `OneHotEncoder` on categorical columns.
- **Regressor**: `LinearRegression`
- **Pipeline**: Built using `sklearn.pipeline.Pipeline`

## Training & Evaluation

- Dataset split: 80% training, 20% testing using `train_test_split`
- Model trained using pipeline
- Evaluation metrics on **training data**:
  - R² Score: **0.66**
  - Mean Absolute Error: **1,429,515 PKR**

## 📈 Prediction Function

A reusable function `predict_price()` takes car attributes as input and returns the predicted price.

Example:
```python
predict_price(
    title='Suzuki Mehran', 
    city='Multan', 
    model=2016,
    mileage=100000,
    fuel_type='Petrol', 
    transmission='Not Available', 
    registered='Multan', 
    color='White', 
    assembly='Local',
    engine_capacity=1000,
    vehicle_age=8
)
# ➞ Predicted Price: 1,216,512 PKR
