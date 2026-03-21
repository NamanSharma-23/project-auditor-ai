import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("dairy_data.csv")

x=df[["Age_Years", "Feed_Kg"]]
y=df["Milk_Liters"]

brain=LinearRegression()
brain.fit(x,y)

test_cow=[[8,10]]
result=brain.predict(test_cow)
print(f"Predicted milk yield: {result[0]:.2f} Liters")

print("--Analysis--")
print(f"Impact of Age: {brain.coef_[0]:.2f}")
print(f"Impact of Feed: {brain.coef_[1]:.2f}")

print(f"Model Accuracy (R-squared): {brain.score(x,y):.2f}")