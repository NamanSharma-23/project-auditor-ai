import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
df=pd.read_csv("project_training.csv")
x=df[["Days","Team_Size"]]
cost_model=LinearRegression()
cost_model.fit(x,df["Actual_Cost"])

time_model=LogisticRegression()
time_model.fit(x,df["On_Time"])

print("🚀 Project Auditor AI: Both brains are trained!")

new_project=[[40,1]]
cost_pred=cost_model.predict(new_project)
time_pred=time_model.predict(new_project)

print(f"--- Results for 40-Day Project ---")
print(f"💰 Predicted Cost: ${cost_pred[0]:,.2f}")

if time_pred[0] == 1:
    print("📅 Prediction: On Track")
else:
    print("⚠️ Prediction: Likely Delayed")