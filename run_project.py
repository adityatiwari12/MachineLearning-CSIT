# Simple Linear Regression Project - Run Script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 1. Load the data
print("Loading data from SALES.txt...")
if not os.path.exists('SALES.txt'):
    print("Error: SALES.txt not found. Please ensure the file exists in the current directory.")
else:
    df = pd.read_csv('SALES.txt', sep='\t', header=None)
    df.columns = ['Sales', 'Advertising']

    # 2. Exploratory Data Analysis
    print("\nDataset Dimensions:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Summary:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())

    # 3. Prepare data
    X = df['Sales'].values.reshape(-1, 1)
    y = df['Advertising'].values.reshape(-1, 1)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 5. Build and Train Model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # 6. Model Parameters
    print(f"\nEstimated model slope (a): {lm.coef_[0][0]:.8f}")
    print(f"Estimated model intercept (b): {lm.intercept_[0]:.8f}")
    print(f"Regression Equation: y = {lm.coef_[0][0]:.8f} * x + {lm.intercept_[0]:.8f}")

    # 7. Predictions and Evaluation
    y_pred = lm.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEvaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Training set score: {lm.score(X_train, y_train):.4f}")
    print(f"Test set score: {lm.score(X_test, y_test):.4f}")

    # 8. Save the model
    joblib.dump(lm, 'lm_regressor.pkl')
    print("\nModel saved as 'lm_regressor.pkl'.")

    # 9. Plotting (Save to file)
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
    plt.title('Sales vs Advertising Relationship')
    plt.xlabel('Sales')
    plt.ylabel('Advertising')
    plt.legend()
    plt.savefig('regression_plot.png')
    print("Regression plot saved as 'regression_plot.png'.")

    plt.figure(figsize=(10, 6))
    plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, color='red', label='Train Residuals')
    plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, color='blue', label='Test Residuals')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.title('Residual Errors')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.savefig('residuals_plot.png')
    print("Residuals plot saved as 'residuals_plot.png'.")

    print("\nProject execution completed successfully.")
