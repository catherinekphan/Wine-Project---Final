#!/usr/bin/env python
# coding: utf-8

# # Red Wine Quality Prediction Project
# 
# ## Introduction
# 
# In this project, we aim to develop a predictive model that assesses the quality of red wine based on its chemical properties. The quality assessment is crucial for wine producers to maintain consistency and meet consumer expectations. By leveraging machine learning techniques, specifically supervised learning algorithms, we seek to predict the quality rating of wine samples on a scale from 1 to 10.
# 
# ### Dataset Overview
# 
# The dataset used in this project contains various chemical properties of red wines, such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. Each entry in the dataset corresponds to a specific wine sample along with its quality rating.
# 
# ### Project Goals
# 
# 1. **Data Preparation**: Cleanse and preprocess the dataset to ensure data quality and uniformity.
# 2. **Exploratory Data Analysis (EDA)**: Explore the dataset to understand relationships between the wine's chemical properties and its quality rating.
# 3. **Feature Engineering**: Identify relevant features and possibly create new ones that enhance predictive performance.
# 4. **Model Selection and Training**: Evaluate various machine learning models suitable for regression tasks and train them on the prepared dataset.
# 5. **Model Evaluation**: Assess the performance of each model using appropriate evaluation metrics to determine the most effective one.
# 6. **Deployment and Future Steps**: Deploy the chosen model for predicting wine quality and outline potential future enhancements or applications.
# 
# ### Methodology
# 
# Our approach will involve dividing the dataset into training and testing sets, training multiple regression models, and evaluating their performance based on metrics such as Mean Squared Error (MSE) and R-squared. Additionally, we will visualize key insights from the data to interpret model predictions and refine our understanding of the factors influencing wine quality.
# 
# By the end of this project, we aim to deliver a robust predictive model that not only accurately predicts red wine quality but also provides valuable insights into the chemical composition factors that contribute to high-quality wines.
# 
# Stay tuned as we embark on this journey to uncover the secrets behind a perfect glass of red wine!

# In[18]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# In[4]:


wine = pd.read_csv("winequality-red.csv")

# Separate features (X) and target (y)
X = wine.drop('quality', axis=1)  # Features are all columns except 'quality'
y = wine['quality']  # Target is the 'quality' column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you have your training and testing sets
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)


# In[22]:


chemical_properties = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                       'pH', 'sulphates', 'alcohol']

# Create a box plot for each chemical property grouped by wine quality
plt.figure(figsize=(12, 8))
for i, property in enumerate(chemical_properties):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x='quality', y=property, data=wine, palette='viridis')
    plt.title(property.capitalize(), fontsize=12)
    plt.xlabel('Quality', fontsize=10)
    plt.ylabel(property.capitalize(), fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

plt.tight_layout()
plt.show()


# Analyzing the distribution of chemical properties across different wine quality ratings through box plots can offer valuable insights into what characteristics contribute to perceived wine quality. Each box plot in this visualization represents how a specific chemical property varies across wine samples of different quality ratings. For instance, observing how 'volatile acidity' or 'sulphates' vary from low to high-quality wines can indicate which chemical attributes might be associated with higher or lower quality ratings. Such visualizations not only aid in identifying potential correlations between chemical composition and wine quality but also inform winemakers and enthusiasts about the key factors that contribute to the overall sensory experience of wine. By understanding these relationships, stakeholders can make informed decisions regarding production processes, ingredient choices, and quality improvement strategies to meet consumer preferences and enhance product quality.

# In[5]:


wine.head()


# In[6]:


X_test.info()


# In[7]:


X_train.info()


# From the information gathered, we observe that our training dataset consists of 1,279 entries, each containing measurements related to various chemical properties of red wine, such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. These attributes serve as our features for predicting the quality of wine, which is represented by the 'quality' column.
# 
# Meanwhile, the test set comprises 320 entries with the same set of features but lacks the 'quality' column, which is the target variable we aim to predict using our trained model.
# 
# Our dataset is well-prepared, with no missing values, ensuring a smooth and reliable analysis process. This cleanliness facilitates accurate modeling and evaluation of our predictive algorithms.
# 
# Our next steps involve conducting thorough Exploratory Data Analysis (EDA) on the training data. EDA will enable us to visualize distributions, identify correlations between features and wine quality, and uncover insights that can guide our model selection and feature engineering efforts.

# In[8]:


# Calculate the distribution of wine quality categories in training data
quality_counts = y_train.value_counts().sort_index()
colors = sns.color_palette("RdPu", len(quality_counts))

# Plotting the distribution using seaborn countplot with custom colors
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train, palette=colors, edgecolor='black')
plt.title("Distribution of Wine Quality Ratings in Training Data", fontsize=16)
plt.xlabel("Wine Quality Rating", fontsize=14)
plt.ylabel("Number of Wines", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(quality_counts):
    plt.text(i, v + 10, str(v), ha='center', fontsize=12)

plt.tight_layout()
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Calculate average values of each feature grouped by 'quality'
avg_values = wine.groupby('quality').mean()

colors = ['salmon', 'plum', 'skyblue', 'lightgreen', 'gold', 'lightcoral']

# Plotting the average values for each quality rating
plt.figure(figsize=(12, 6))
for i, quality in enumerate(avg_values.index):
    plt.bar(avg_values.columns, avg_values.loc[quality], label=f'Quality {quality}', color=colors[i], alpha=0.7)

plt.title("Average Chemical Properties by Wine Quality Rating", fontsize=16)
plt.xlabel("Chemical Properties", fontsize=14)
plt.ylabel("Average Value", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()


# The use of matplotlib.pyplot to plot average chemical properties by wine quality rating plays a crucial role in the Exploratory Data Analysis (EDA) phase of our project. By grouping the dataset by wine quality and calculating average values of key chemical properties, we gain valuable insights into how these properties vary across different quality ratings. This visualization aids in understanding the relationships between the input features (chemical properties) and the target variable (wine quality rating), which is fundamental for our subsequent modeling efforts. Specifically, this step helps us identify which chemical properties might have a significant impact on wine quality and informs potential feature engineering strategies. As we visualize and interpret these average values, we are actively engaged in exploring patterns and preparing our dataset for the next stages of model selection, training, and evaluation.
# 
# Currently, we are in the Exploratory Data Analysis (EDA) phase of the project, where we are exploring the dataset to understand relationships between the wine's chemical properties and its quality rating.

# Feature Training

# In[12]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[13]:


print("Number of features in X_train_scaled:", X_train_scaled.shape[1])
print("Number of features in X_test_scaled:", X_test_scaled.shape[1])


# In preparation for modeling wine quality prediction, feature scaling is applied to the numerical attributes of the dataset using StandardScaler from scikit-learn. This process standardizes each feature by removing the mean and scaling to unit variance, ensuring all variables contribute equally to model training without bias from differing scales. After scaling, the transformed NumPy array (X_train_scaled) is converted back into a DataFrame format, retaining the original column names from X_train. Leveraging domain knowledge, a new feature called 'total_acidity' is engineered by summing the fixed acidity and volatile acidity columns from the original dataset. This engineered feature aims to capture a combined acidity measure, potentially enhancing the predictive capability of subsequent machine learning models. This iterative approach of scaling, DataFrame conversion, and feature engineering aligns with best practices in preprocessing data to optimize model performance and interpretability.

# In[16]:


best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train_scaled, y_train)


# In[19]:


y_pred = best_model.predict(X_test_scaled)

# Evaluate performance metrics
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score on Test Set: {r2:.3f}")


# In[20]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation
scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-validated R^2 Score: {scores.mean():.3f} (± {scores.std():.3f})")


# ### Explanation of ** R-Squared Score and Cross-Validation**
# 
# The R-Squared score, or coefficient of determination, is pivotal in regression tasks like predicting wine quality using machine learning models. It quantifies how much of the variance in the dependent variable (wine quality) can be explained by the independent variables (chemical properties of wine). Achieving an R-Squared score of 0.540 on the test set suggests that the RandomForestRegressor model can account for 54.0% of the variability in wine quality. This indicates a moderate level of predictive capability, implying that the model captures a significant portion of the variation in wine quality based on the provided features.
# 
# **Cross-validation** enhances the reliability of model evaluation by testing its performance across multiple subsets of the training data. The cross-validated R-Squared score of 0.426 (± 0.067) signifies that the model's predictions remain consistent across different folds of the training data. This range (± 0.067) provides insights into the model's stability and its ability to generalize to unseen data. A higher cross-validated R-Squared score indicates that the model's performance is robust and reliable, suggesting it can generalize well beyond the specific test set used.
# 
# ### Conclusion
# 
# In summary, the RandomForestRegressor demonstrates promising predictive performance with an R-Squared score of 0.540 on the test set. The cross-validated score of 0.426 (± 0.067) further underscores its reliability and consistency. Together, these metrics indicate that the model can make reasonably accurate predictions of wine quality based on the selected chemical properties. Future enhancements could involve additional feature engineering or exploring alternative algorithms to potentially further improve the model's predictive power in practical applications of wine quality prediction.
# 
