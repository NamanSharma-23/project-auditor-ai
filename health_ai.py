import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("health_data.csv")
x=df[["Temperature"]]
y=df["Status"]

clf=LogisticRegression()
clf.fit(x,y)

new_temp=[[33]]
prediction=clf.predict(new_temp)

if prediction[0]==1:
    print("✅ The AI says: Cow is Healthy.")
else:
    print("⚠️ The AI says: Cow needs a Checkup!")

print(f"Model Accuracy: {clf.score(x,y):.2f}")
prob=clf.predict_proba(new_temp)
print(f"Confidence: {prob[0][1]*100:.1f}% chance of being Healthy")