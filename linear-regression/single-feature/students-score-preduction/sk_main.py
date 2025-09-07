import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("study_scores.csv")
print(df.head())

x = df["Hours_Studied"].values
y = df["Exam_Score"].values


plt.scatter(x, y, color="red", marker="*")
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = [[1]]
print("Predicted Score:", model.predict(y_pred))

plt.scatter(x, y, color="red", marker="*")
plt.plot(x, model.predict(x.reshape(-1, 1)), color="blue")
plt.show()