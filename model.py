import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Function to create and train the model
def train_model():
    # Sample data: hours studied, attendance rate, final score
    data = {
        "hours_studied": [10, 20, 30, 40, 50],
        "attendance_rate": [80, 85, 88, 90, 95],
        "final_score": [75, 82, 85, 88, 90]
    }
    df = pd.DataFrame(data)

    # Prepare the data
    X = df[['hours_studied', 'attendance_rate']]  # Features
    y = df['final_score']  # Target

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    print("Model trained successfully!")
    return model

if __name__ == "__main__":
    train_model()
