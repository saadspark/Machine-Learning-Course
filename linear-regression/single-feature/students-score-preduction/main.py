import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("study_scores.csv")

X = df["Hours_Studied"].values
y = df["Exam_Score"].values

# Initialize parameters
w = 0.0
b = 0.0
lr = 0.001   # learning rate
epochs = 1000
n = len(X)

mse_history = []

# Gradient Descent Loop
for epoch in range(epochs):
    # Predictions
    y_pred = w * X + b
    
    # Error
    error = y_pred - y
    
    # MSE
    mse = (error**2).mean()
    mse_history.append(mse)
    
    # Gradients
    dw = (2/n) * (X * error).sum()
    db = (2/n) * error.sum()
    
    # Update parameters
    w -= lr * dw
    b -= lr * db

# Final predictions
y_pred = w * X + b

# Plot regression line
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# Plot loss curve
plt.plot(mse_history)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Loss Curve (MSE over Epochs)")
plt.show()

print(f"Trained Slope (w): {w:.2f}")
print(f"Trained Intercept (b): {b:.2f}")
print(f"Final MSE: {mse:.2f}")
