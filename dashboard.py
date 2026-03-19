import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🚀 Naman's AI Project Auditor")
st.write("This dashboard automatically analyzes project risks and budget.")

df=pd.read_csv("projects.csv")
df["Budget"]=df["Budget"].fillna(0)

def analyze_risk(budget):
    if budget > 10000: return "🔴 High Risk"
    if budget > 5000: return "🟡 Medium Risk"
    return "🟢 Low Risk"

df["Risk_Level"]=df["Budget"].apply(analyze_risk)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Projects", len(df))

with col2:
    total_budget=df["Budget"].sum()
    st.metric("Total Budget", f"${total_budget:,.0f}")

with col3:
    high_risk_count=len(df[df["Risk_Level"]=="🔴 High Risk"])
    st.metric("High Risk Alerts", high_risk_count, delta="- Action Required", delta_color="inverse")

st.divider()

st.subheader("Project Data Overview")
st.dataframe(df, use_container_width=True)

st.subheader("Budget Visualizer")
fig, ax=plt.subplots()
ax.bar(df["Project"],df["Budget"],color='skyblue')
st.pyplot(fig)

st.divider()
selected_project = st.selectbox("Select a project to audit: ", df["Project"])
project_info=df[df["Project"]==selected_project]
st.write(f"The AI suggests: **{project_info['Risk_Level'].values[0]}**")