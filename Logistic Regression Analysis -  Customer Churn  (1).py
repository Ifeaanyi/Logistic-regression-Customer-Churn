#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the neccessary library for EDA
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing my file into pandas and reading it with a dataframe
file_path = r"C:\Users\USER\Desktop\Adika\Customer Churn Logistics regression project\Churn_Modelling.csv"
df =  pd.read_csv(file_path)
#Displaying my data in a dataframe
df.head()


# In[3]:


#checking for null values
df.isnull().sum()


# In[4]:


#checking for duplicates
df.duplicated()


# In[5]:


#renamin columns
df = df.rename(columns ={'IsActiveMember' : 'Active membership', 'Exited' : 'Churn', 'HasCrCard' : 'Owns a credit card'})


# In[6]:


df.head()


# In[7]:


#checking the age distribution using a histogram
column_name = 'Age'

#create a histogram
plt.figure(figsize=(8,5))
plt.hist(df[column_name], bins = 20, color = 'skyblue', edgecolor ='black')
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.title(f'Distribution of {column_name}')
plt.grid(True)
plt.show()


# In[8]:


#create a box plot to visualise the distribution of estimated salary by gender
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x = 'Gender', y = 'EstimatedSalary', palette = 'Set1')

#setting the plot labels and title
plt.xlabel('Gender')
plt.ylabel('EstimatedSalary')
plt.title('Estimated salary distribution by gender')

#show plot
plt.show()


# In[9]:


#Gender distribution 
plt.figure(figsize=(8,5))

#create bar chart 
sns.countplot(data=df, x = 'Gender', palette = 'Set1')

#set plot labels and titles
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')

plt.show()


# In[10]:


#creating a count plot to visualise gender and churn 
plt.figure(figsize=(8,5))
sns.countplot(data=df, x = 'Gender', hue = 'Churn', palette = 'Set1')

#setting plot and label
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Churn Distribution by Gender')

#adding a legend
plt.legend(title='Churn', labels= ['No churn', 'Churn'] )

#show plot
plt.show()


# In[11]:


#creating a histogram for balance distribution
plt.figure(figsize=(8,5))
plt.hist(df['Balance'], bins = 30, color = 'blue', alpha = 0.7)
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.title('Histogram of balance')
plt.grid(axis = 'y', linestyle = '--')

#show plot
plt.show()


# In[12]:


# excluding non-numeric columns from the correlation calculation
numeric_columns = df.select_dtypes(include=[np.number])
corr_matrix = numeric_columns.corr()

# Display the correlation matrix as a table with a background gradient and desired precision
corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}")


# In[13]:


#selecting feature(Independend) and target (Dependent) variable
x = df[['Age', 'Balance']]
y = df['Churn']


# In[14]:


#importing libraries needed to train and test the data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[15]:


#splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#creating a logistic regression model
model = LogisticRegression()

#training the model
model.fit(x_train, y_train)

#make predictions from the test data
y_pred = model.predict(x_test)

#calculating the accuracy of my model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[20]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Create a confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix with a heatmap
plt.imshow(confusion, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks([0, 1], ['Not Churn', 'Churn'])
plt.yticks([0, 1], ['Not Churn', 'Churn'])

# Display values in each cell of the matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(confusion[i, j]), horizontalalignment='center', verticalalignment='center', color='white')

plt.show()



# In[23]:


# Get predicted probabilities for both class 0 and 1
predicted_probabilities = model.predict_proba(x_test)

# Extract probabilities for the positive churn (1)
positive_class_probabilities = predicted_probabilities[:, 1]

# Extract probabilities for the negative churn (0)
negative_class_probabilities = predicted_probabilities[:, 0]

# Print the predicted probabilities for both classes
print("Predicted Probabilities for Positive Class (Churn):")
print(positive_class_probabilities)

print("\nPredicted Probabilities for Negative Class (No Churn):")
print(negative_class_probabilities)




# In[ ]:




