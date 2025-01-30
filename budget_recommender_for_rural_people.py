#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

np.random.seed(42)
num_records = 500

categories = ["Groceries", "Utilities", "Transportation", "Healthcare", "Entertainment", "Savings"]

income = np.random.randint(5000, 15001, size=num_records)

groceries_prop = np.random.uniform(0.3, 0.6, size=num_records)
utilities_prop = np.random.uniform(0.02, 0.06, size=num_records)
transport_prop = np.random.uniform(0.03, 0.08, size=num_records)
healthcare_prop = np.random.uniform(0.01, 0.05, size=num_records)
entertainment_prop = np.random.uniform(0.02, 0.06, size=num_records)
savings_prop = np.random.uniform(0.05, 0.15, size=num_records)

groceries = groceries_prop * income
utilities = utilities_prop * income
transport = transport_prop * income
healthcare = healthcare_prop * income
entertainment = entertainment_prop * income
savings = savings_prop * income

df = pd.DataFrame({
    "Income": income,
    "Groceries": groceries.round(2),
    "Utilities": utilities.round(2),
    "Transportation": transport.round(2),
    "Healthcare": healthcare.round(2),
    "Entertainment": entertainment.round(2),
    "Savings": savings.round(2)
})

df["Total Expenses"] = df[["Groceries", "Utilities", "Transportation", "Healthcare", "Entertainment"]].sum(axis=1)

df["Grocery Budget"] = 0.4 * df["Income"]

df["Overspending"] = df["Groceries"] > df["Grocery Budget"]

print(df.head(10))



# In[3]:


df.head(10)


# In[5]:


df.size


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(15, 12))


# In[11]:


plt.subplot(1,1,1)
sns.histplot(df['Income'],kde=True,bins=20,color='blue')
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")


# In[13]:


plt.subplot(1,1,1)
sns.histplot(df['Groceries'], kde=True, bins=20, color='green')
plt.title("Groceries Distribution")
plt.xlabel("Groceries")
plt.ylabel("Frequency")


# In[15]:


plt.subplot(1, 1, 1)
sns.scatterplot(x='Income', y='Total Expenses', hue='Overspending', data=df, palette='coolwarm')
plt.title("Income vs Total Expenses")
plt.xlabel("Income (₹)")
plt.ylabel("Total Expenses (₹)")


# In[17]:


plt.subplot(1,1,1)
sns.boxplot(x='Overspending', y='Groceries', data=df, palette='Set2')
plt.title("Grocery Expenses by Overspending")
plt.xlabel("Overspending (True/False)")
plt.ylabel("Groceries")


# In[19]:


plt.subplot(1, 1, 1)
corr_cols = ['Income','Groceries','Utilities','Transportation','Healthcare','Entertainment','Savings','Total Expenses']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='viridis', fmt=".6f", square=True)
plt.title("Correlation Matrix")


# In[21]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[23]:


df.info()


# In[25]:


df.describe()


# In[27]:


df.head()


# In[29]:


X = df.drop(['Overspending', 'Total Expenses', 'Grocery Budget'], axis=1)
y = df['Overspending']


# In[31]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)


# In[33]:


X_train.shape


# In[35]:


X_test.shape


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[39]:


pipelines = {
    'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='liblinear'))]),
    'RandomForest': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))]),
    'GradientBoosting': Pipeline([('scaler', StandardScaler()), ('clf', GradientBoostingClassifier(random_state=42))]),
    'AdaBoost': Pipeline([('scaler', StandardScaler()), ('clf', AdaBoostClassifier(random_state=42))]),
    'SVC': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
    'DecisionTree': Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=42))])
}


# In[41]:


param_grids = {
    'LogisticRegression': {'clf__C': [0.01, 0.1, 1, 10]},
    'RandomForest': {'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 5, 7]},
    'GradientBoosting': {'clf__learning_rate': [0.01, 0.1], 'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 5]},
    'AdaBoost': {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.1]},
    'SVC': {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']},
    'DecisionTree': {'clf__max_depth': [3, 5, 7]}
}


# In[43]:


best_model = None
best_model_name = None
best_accuracy = 0.0


# In[45]:


for model_name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grids[model_name], scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    cv_accuracy = grid_search.best_score_
    
    print(model_name)
    print(f"Best CV Accuracy: {cv_accuracy:.4f}")
    print("Best Hyperparameters:", grid_search.best_params_, "\n")
    
    if cv_accuracy > best_accuracy:
        best_accuracy = cv_accuracy
        best_model = grid_search.best_estimator_
        best_model_name = model_name


# In[47]:


print(f"Overall Best Model: {best_model_name} with CV Accuracy: {best_accuracy:.4f}\n")


# In[49]:


y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with {best_model_name}: {test_acc:.4f}\n")
print("=Classification Report=")
print(classification_report(y_test, y_pred))


# In[51]:


import pickle 


# In[53]:


pickle_filename = "best_model.pkl"


# In[55]:


with open(pickle_filename,'wb') as file:
    pickle.dump(best_model, file)


# In[ ]:




