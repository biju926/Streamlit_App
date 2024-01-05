#!/usr/bin/env python
# coding: utf-8

# In[210]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency 


# In[211]:


df1 = pd.read_csv("churn-bigml-80.csv")
df.head()


# In[212]:


df2 = pd.read_csv("churn-bigml-20.csv")
df.head()


# In[213]:


df1.shape


# In[214]:


df2.shape


# In[215]:


df = pd.concat([df1,df2], axis = 0)
df.head()


# In[216]:


df.shape


# In[31]:


df.info()


# In[32]:


df.isnull().sum()


# In[33]:


df.describe()


# In[136]:


df.describe(include=["object"]).T


# In[60]:


churn =df['Churn'].value_counts()
churn


# In[160]:


plt.pie(churn,labels=labels,autopct='%1.1f%%')
plt.title('Overall Count of Churn')
plt.legend()
plt.show()


# <b>It is very clear from the above figure that no. of customers churned is very less.

# In[156]:


churn_intl_plan=df.groupby(["Churn","International plan"]).count()[["Area code"]]
churn_intl_plan.rename(columns={"Area code":"Count"},inplace=True)
churn_intl_plan.head(50)


# In[166]:


# Percentage of customers with international plan who has churned with respect to who dont have plan
Churn_true_intl_plan_percentage = round(137*100/(346+137),2)
Churn_true_intl_plan_percentage


# In[169]:


# Percentage of customers international plan who has not churned with respect to who dont have plan
Churn_false_intl_plan_percentage = round(186*100/(2664+186),2)
Churn_false_intl_plan_percentage


# <b>If we look into the above data it is very clear that the customer who churn has an high percentage of International plan with respect to those who doesnt have international plan when compared with customers who doesn't churn

# In[ ]:





# #### Analysis of Categorical columns with target column Churn

# In[170]:


fig,ax=plt.subplots(1,1,figsize=(6,6))
sns.countplot(x='International plan',data=df,hue='Churn',)
plt.title('Churn Count by International Plan')
plt.show()


# In[161]:


# chi-square test for two categorical data
def chi_2_test(df, col2,col1 = "Churn"):
    data = pd.crosstab(df[col1], df[col2], rownames=[col1], colnames=[col2])
    stat, p, dof, expected = chi2_contingency(data)
    alpha = 0.05
    print("p value is " + str(p))
    if p <= alpha:
        print(f"{col2} and {col1} are dependant!")
    else : 
        print(f"{col2} and {col1} are independant!")


# In[88]:


chi_2_test(df,"International plan")


# In[171]:


sns.countplot(data = df, x = "Voice mail plan", hue = "Churn")
plt.title('Churn Count by Voice Mail Plan')


# In[92]:


chi_2_test(df,"Voice mail plan")


# In[173]:


sns.countplot(data = df, x = "Area code", hue = "Churn")
plt.title('Churn Count by Area code')


# In[175]:


chi_2_test(df,"Area code")


# #### From the above figure it is very clear that most of the loyal custoners belong to Area code 415 and also p value says that Area code has nothing to do with churn

# In[241]:


fig, axes = plt.subplots(10,5,figsize = (20,30))
for i in range(len(df.State.unique())-1): 
    # filter for that state 
    state_data = df[df.State == df.State.unique()[i]]
    sns.countplot(data = state_data, x = "State", hue = "Churn", ax = axes[i//5,i%5])
    axes[i//5,i%5].bar_label(axes[i//5,i%5].containers[1])
    axes[i//5,i%5].bar_label(axes[i//5,i%5].containers[0])
    axes[i//5,i%5].set_ylabel("")
    axes[i//5,i%5].set_xlabel("")


# #### From the above charts we can see that some of the states such as SC,MS,MI,NV,ME,CA, MD and NJ has more churn rate

# In[247]:


df2=df.groupby(["State","Churn"]).count()[["Area code"]]
df2.head(49)


# #### Mean, Median, standard Deviation, Min and Max for total minutes for Day, Evening, Night and International

# In[259]:


minutes_columns = ["Total day minutes", "Total eve minutes", "Total night minutes","Total intl minutes"]
df.groupby(["Churn"])[minutes_columns].describe().T


# #### Mean, Median, standard Deviation, Min and Max for total charges for Day, Evening, Night and International

# In[258]:


charge_columns = ["Total day charge", "Total eve charge", "Total night charge","Total intl charge"]
df.groupby(["Churn"])[charge_columns].describe().T


# #### Mean, Median, standard Deviation, Min and Max for total calls for Day, Evening, Night and International

# In[256]:


call_columns = ["Total day calls", "Total eve calls", "Total night calls","Total intl calls"]
df.groupby(["Churn"])[call_columns].describe().T


# In[145]:


df["Total calls"]= df["Total day calls"] + df["Total eve calls"] + df["Total night calls"]+ df["Total intl calls"]


# In[146]:


df.head()


# In[147]:


df["Total charge"] =df["Total day charge"] + df["Total eve charge"] + df["Total night charge"] + df["Total intl charge"]


# In[148]:


df.head()


# In[ ]:





# #### Analysis of Numerical columns with Target column Churn

# In[198]:


# boxplot analysis of Total calls for day, evening, night and internation calls
fig, axes = plt.subplots(2,2,figsize = (10,8))
sns.boxplot(  x= "Churn", y= "Total day calls", data=df , ax=axes[0,0])
sns.boxplot(  x= "Churn", y= "Total eve calls", data=df , ax=axes[0,1])
sns.boxplot(  x= "Churn", y= "Total night calls", data=df , ax=axes[1,0])
sns.boxplot(  x= "Churn", y= "Total intl calls", data=df , ax=axes[1,1])
plt.show()


# #### When it comes to calls only international call have some relation with churn. Most of the customers who churns have no. of international call less than the average call.

# In[199]:


# boxplot analysis of Total charges for day, evening, night and internation calls
fig, axes = plt.subplots(2,2,figsize = (10,8))
sns.boxplot(  x= "Churn", y= "Total day charge", data=df , ax=axes[0,0])
sns.boxplot(  x= "Churn", y= "Total eve charge", data=df , ax=axes[0,1])
sns.boxplot(  x= "Churn", y= "Total night charge", data=df , ax=axes[1,0])
sns.boxplot(  x= "Churn", y= "Total intl charge", data=df , ax=axes[1,1])
plt.show()


# In[200]:


# boxplot analysis of Total minutes for day, evening, night and internation calls
fig, axes = plt.subplots(2,2,figsize = (10,8))
sns.boxplot(  x= "Churn", y= "Total day minutes", data=df , ax=axes[0,0])
sns.boxplot(  x= "Churn", y= "Total eve minutes", data=df , ax=axes[0,1])
sns.boxplot(  x= "Churn", y= "Total night minutes", data=df , ax=axes[1,0])
sns.boxplot(  x= "Churn", y= "Total intl minutes", data=df , ax=axes[1,1])
plt.show()


# In[238]:


fig=plt.subplots(1,1,figsize=(12,8))
sns.boxplot(data=df, x="International plan", y="Total intl charge", hue="Churn")
plt.show()


# ##### From the above figure it is very clear that if a customer has an international plan then he/she may churn since the average international charges are high when compared with customers who doesnt churn 

# In[253]:


fig=plt.subplots(1,1,figsize=(12,8))
sns.boxplot(data=df, y="Account length", x="Churn")
plt.show()


# In[ ]:




