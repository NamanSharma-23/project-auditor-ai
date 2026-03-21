import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle
df=pd.read_csv("project_training.csv")
x=df[["Days","Team_Size"]]

cost_model=LinearRegression().fit(x,df["Actual_Cost"])
time_model=LogisticRegression().fit(x,df["On_Time"])

with open("cost_model.pkl", "wb") as f:
    pickle.dump(cost_model,f)
with open("time_model.pkl","wb") as f:
    pickle.dump(time_model,f)

print("✅ Models saved as .pkl files!")