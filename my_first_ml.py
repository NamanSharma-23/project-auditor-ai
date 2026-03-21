import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("simple_data.csv")

x=data[["Weight", "Quality"]]
y=data["Price"]

brain=LinearRegression()
brain.fit(x,y)

new_item=[[10,8]]
result = brain.predict(new_item)
print(f"For 10 kg with quality 8 AI predicts: ${result[0]}")

print(f"Weights assigned by AI: {brain.coef_}")
print(f"The 'Starting Point' (Intercept): {brain.intercept_}")