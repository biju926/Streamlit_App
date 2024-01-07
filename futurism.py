#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency 


st.title("Telecom Churn Dataset")
st.markdown(
    """About Dataset\n
Context \n
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs."

Content\n
The Orange Telecom's Churn Dataset, which consists of cleaned customer activity data (features), along with a churn label specifying whether a customer canceled the subscription, will be used to develop predictive models. Two datasets are made available here: The churn-80 and churn-20 datasets can be downloaded.

"""
)

df1 = pd.read_csv("churn-bigml-80.csv")
#df1.head()

df2 = pd.read_csv("churn-bigml-20.csv")
#df2.head()

df = pd.concat([df1,df2], axis = 0)
#df.head()
tab1, tab2, tab3 = st.tabs(["Dataset1(80)", "Dataset2(20)", "Combined Dataset"])
with tab1:
  st.write("Head of the dataset(80) ")
  st.dataframe(df1.head())
with tab2:
  st.write("Head of the dataset(20) ")
  st.write(df2.head())
with tab3:
  st.write("Head of the combined dataset")
  st.write(df.head())


churn =df['Churn'].value_counts()
fig, axes = plt.subplots(figsize = (5,5))
label =["No","Yes"]
plt.pie(churn,labels = label,autopct='%1.1f%%')
plt.title('Overall churn in terms of percentage')

with st.sidebar:
    tab1, tab2, tab3 = st.tabs(["Dataset1", "Dataset2", "Combined Dataset"])
    with tab1:
      st.write(""" The Number of rows  and columns of dataset 1 """)
      st.write(df1.shape)
    with tab2:
      st.write(df2.shape)
    with tab3:
        st.write(df.shape)
        

with st.sidebar:
    tab1, tab2 = st.tabs(["📈 Chart of Churn", "🗃 Churn Data"])
    with tab1:
      st.pyplot(fig)
    with tab2:
      st.write("14.5 % customers have Churned.")
      st.write(churn)
    

df['Churn']=df["Churn"].map({True:1,False:0})
df['International plan']=df['International plan'].map({"Yes":1,"No":0})
df['Voice mail plan']=df['Voice mail plan'].map({"Yes":1,"No":0})

#st.write(df.info())
st.write("**The count of Null values in the dataset**")
st.write(df.isnull().sum())
st.write("**No null values found in the dataset**")
#df.describe(include=["object"]).T

churn_intl_plan=df.groupby(["Churn","International plan"]).count()[["Area code"]]
churn_intl_plan.rename(columns={"Area code":"Count"},inplace=True)
churn_intl_plan.head()

width = st.sidebar.slider("plot width", 1, 25, 3)
height = st.sidebar.slider("plot height", 1, 25, 1)


fig1, axes = plt.subplots(figsize = (10,6))
sns.countplot(data = df, x = "International plan", hue = "Churn")
plt.title('Churn Count by ,International Plan')


fig2, axes = plt.subplots(figsize = (width,height))
sns.countplot(data = df, x = "Voice mail plan", hue = "Churn")
plt.title('Churn Count by Voice Mail Plan')


fig3, axes = plt.subplots(figsize = (10,6))
sns.countplot(data = df, x = "Area code", hue = "Churn")
plt.title('Churn Count by Area code')


st.write("**Analysis of Categorical columns with target column Churn**")

# chi-square test for two categorical data
def chi_2_test(df, col2,col1 = "Churn"):
    data = pd.crosstab(df[col1], df[col2], rownames=[col1], colnames=[col2])
    stat, p, dof, expected = chi2_contingency(data)
    alpha = 0.05
    st.write("p value is ", p)
    if p <= alpha:
        st.write(f"{col2} and {col1} are dependant!")
    else : 
        st.write(f"{col2} and {col1} are independant!")

tab1, tab2, tab3 = st.tabs(["Intl Plan Vs Churn", "Voice Mail Plan Vs Churn", "Area Code Vs Churn"])
with tab1:
  st.pyplot(fig1)
  chi_2_test(df,"International plan")
  # Percentage of customers with international plan who has churned with respect to who dont have plan
  Churn_true_intl_plan_percentage = round(137*100/(346+137),2)
  st.write("Percentage of customer churn having international plan",Churn_true_intl_plan_percentage)
  Churn_false_intl_plan_percentage = round(186*100/(2664+186),2)
  # Percentage of customers international plan who has not churned with respect to who dont have plan
  st.write("Percentage of customer doesnt churn having international plan",Churn_false_intl_plan_percentage)
  with st.expander("See explanation"):
    st.write("""
        If we look into the above data it is very clear that the customer who churn has an high percentage of International plan with respect to those who doesnt have international plan when compared with customers who doesn't churn
    """)
with tab2:
  st.pyplot(fig2)
  chi_2_test(df,"Voice mail plan")
  with st.expander("See explanation"):
    st.write("""
        Since the p-values is less than 0.05 which is significant to say that Voice mail plan has dependecy with churn
    """)
with tab3:
  st.pyplot(fig3)
  chi_2_test(df,"Area code")
  with st.expander("See explanation"):
    st.write("""
        From the above figure it is very clear that most of the loyal custoners belong to Area code 415 and also p value says that Area code has nothing to do with churn.
    """)


st.write("**Analysis of States with respect to Churn**")
fig, axes = plt.subplots(10,5,figsize = (20,30))
for i in range(len(df.State.unique())-1): 
    # filter for that state 
    state_data = df[df.State == df.State.unique()[i]]
    sns.countplot(data = state_data, x = "State", hue = "Churn", ax = axes[i//5,i%5])
    axes[i//5,i%5].bar_label(axes[i//5,i%5].containers[1])
    axes[i//5,i%5].bar_label(axes[i//5,i%5].containers[0])
    axes[i//5,i%5].set_ylabel("")
    axes[i//5,i%5].set_xlabel("")
st.pyplot(fig)
with st.expander("See explanation"):
    st.write("""
        From the above charts we can see that some of the states such as SC,MS,MI,NV,ME,CA, MD and NJ has more churn rate.
    """)

df2=df.groupby(["State","Churn"]).count()[["Area code"]]
df2.head(49)

st.write("**Statistical data of total minutes for Day, Evening, Night and International**")
tab1, tab2, tab3, tab4 = st.tabs(["Day Minutes", "Evening Minutes", "Night Minutes", "International Minutes"])
with tab1:
  st.dataframe(df.groupby(["Churn"])["Total day minutes"].describe())
with tab2:
  st.dataframe(df.groupby(["Churn"])["Total eve minutes"].describe())
with tab3:
  st.dataframe(df.groupby(["Churn"])["Total night minutes"].describe())
with tab4:
  st.dataframe(df.groupby(["Churn"])["Total intl minutes"].describe())

st.write("**Statistical data of total Charge for Day, Evening, Night and International**")
tab1, tab2, tab3, tab4 = st.tabs(["Day Charge", "Evening Charge", "Night Charge", "International Charge"])
with tab1:
  st.dataframe(df.groupby(["Churn"])["Total day charge"].describe())
with tab2:
  st.dataframe(df.groupby(["Churn"])["Total eve charge"].describe())
with tab3:
  st.dataframe(df.groupby(["Churn"])["Total night charge"].describe())
with tab4:
  st.dataframe(df.groupby(["Churn"])["Total intl charge"].describe())

st.write("**Statistical data of total Calls for Day, Evening, Night and International**")
tab1, tab2, tab3, tab4 = st.tabs(["Day Calls", "Evening Calls", "Night Calls", "International Calls"])
with tab1:
  st.dataframe(df.groupby(["Churn"])["Total day calls"].describe())
with tab2:
  st.dataframe(df.groupby(["Churn"])["Total eve calls"].describe())
with tab3:
  st.dataframe(df.groupby(["Churn"])["Total night calls"].describe())
with tab4:
  st.dataframe(df.groupby(["Churn"])["Total intl calls"].describe())


df["Total calls"]= df["Total day calls"] + df["Total eve calls"] + df["Total night calls"]+ df["Total intl calls"]

df.head()

df["Total charge"] =df["Total day charge"] + df["Total eve charge"] + df["Total night charge"] + df["Total intl charge"]

df.head()

st.write("**Analysis of Numerical columns with Target column Churn**")
fig3, axes = plt.subplots(2,2,figsize = (10,8))
st.write("**boxplot analysis of Total calls for day, evening, night and internation calls**")

sns.boxplot(  x= "Churn", y= "Total day calls", data=df , ax=axes[0,0])
sns.boxplot(  x= "Churn", y= "Total eve calls", data=df , ax=axes[0,1])
sns.boxplot(  x= "Churn", y= "Total night calls", data=df , ax=axes[1,0])
sns.boxplot(  x= "Churn", y= "Total intl calls", data=df , ax=axes[1,1])
plt.show()
st.pyplot(fig3)
with st.expander("See explanation"):
    st.write("""
        When it comes to calls only international call have some relation with churn. Most of the customers who churns have no. of international call less than the average call.
    """)

st.write("**boxplot analysis of Total charges for day, evening, night and internation calls**")
fig4, axes = plt.subplots(2,2,figsize = (10,8))
sns.boxplot(  x= "Churn", y= "Total day charge", data=df , ax=axes[0,0])
sns.boxplot(  x= "Churn", y= "Total eve charge", data=df , ax=axes[0,1])
sns.boxplot(  x= "Churn", y= "Total night charge", data=df , ax=axes[1,0])
sns.boxplot(  x= "Churn", y= "Total intl charge", data=df , ax=axes[1,1])
plt.show()
st.pyplot(fig4)
with st.expander("See explanation"):
    st.write("""
        When it comes to charges only day call have some relation with churn. Most of the customers who churns high average call charges when compares to customers who dont churn.
    """)

st.write("**boxplot analysis of Total minutes for day, evening, night and internation calls**")
fig5, axes = plt.subplots(2,2,figsize = (10,8))
sns.boxplot(  x= "Churn", y= "Total day minutes", data=df , ax=axes[0,0])
sns.boxplot(  x= "Churn", y= "Total eve minutes", data=df , ax=axes[0,1])
sns.boxplot(  x= "Churn", y= "Total night minutes", data=df , ax=axes[1,0])
sns.boxplot(  x= "Churn", y= "Total intl minutes", data=df , ax=axes[1,1])
plt.show()
st.pyplot(fig5)
with st.expander("See explanation"):
    st.write("""
        When it comes to minutes only day call have some relation with churn. Most of the customers who churns high average call minutes when compares to customers who dont churn.
    """)

st.write("**boxplot analysis of International plan and International Charge with respecrt to Churn**")
fig6, axes=plt.subplots(1,1,figsize=(12,8))
sns.boxplot(data=df, x="International plan", y="Total intl charge", hue="Churn")
plt.show()
st.pyplot(fig6)
with st.expander("See explanation"):
    st.write("""
        From the above figure it is very clear that if a customer has an international plan then he/she may churn since the average international charges are high when compared with customers who doesnt churn .
    """)
   
st.write("**boxplot analysis of Total minutes for day, evening, night and internation calls**")
fig7, axes=plt.subplots(1,1,figsize=(12,8))
sns.boxplot(data=df, y="Account length", x="Churn")
plt.show()
st.pyplot(fig7)
with st.expander("See explanation"):
    st.write("""
        From the above figure it is very clear that Account length has nothing to do with Churn.
    """)

