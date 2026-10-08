# Sales Forecasting

## Project Overview

This project predicts future sales using historical sales data.

## Objective

The main objective is to analyze historical sales and build a machine learning model to forecast sales.

## Dataset

The project uses the Superstore Sales dataset.

The dataset contains information such as:

- Order Date
- Sales
- Product
- Category
- Region
- Customer information

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Model

Random Forest Regressor

## Project Workflow

1. Load the sales dataset
2. Convert Order Date into datetime format
3. Sort data by date
4. Aggregate sales monthly
5. Create lag features
6. Split data into training and testing sets
7. Train Random Forest model
8. Predict sales
9. Evaluate the model
10. Visualize actual vs predicted sales

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Output

The project generates:

- Monthly Sales Trend graph
- Actual vs Predicted Sales graph

## Project Structure

```text
Sales-Forecasting/
│
├── Sample - Superstore.csv
├── sales_forecasting.py
├── requirements.txt
├── README.md
├── monthly_sales_trend.png
├── actual_vs_predicted_sales.png
└── output/