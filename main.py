# Import necessary libraries
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import linalg as LA
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load data from an Excel file
df = pd.read_excel('BClearning.xlsx')
print(df)

# Drop the 'id' column from the dataset
df = df.drop("id",axis = 1)
print(df)

# Separate the target variable ('B.C.') from the feature variables
y = df.loc[:,["B.C."]].values
x = StandardScaler().fit_transform(df.drop(columns=["B.C."]))

# Scale the numeric columns to have a z-score of 0
df_numeric = df.select_dtypes(include=['int', 'float'])
df_zscore = df_numeric.apply(zscore)

# Compute the covariance matrix for the z-scored data
cov_matrix = df_zscore.cov()
print(cov_matrix)

# Perform principal component analysis (PCA) to reduce the dimensionality of the data to 3 components
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(df_zscore)

# Create a new dataframe with the principal components and the target variable
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])
finalDf = pd.concat([principalDf, df[['B.C.']]], axis = 1)

# Create a 3D scatter plot to visualize the data
fig = px.scatter_3d(finalDf, x='PC1', y='PC2',z = "PC3", color='B.C.')
fig.show()

# Compute the mean of the z-scored data
df_mean = df_zscore.mean(numeric_only=True)
print(df_mean)

# Compute the eigenvalues and eigenvectors of the covariance matrix
w,v= LA.eig(cov_matrix)
print(w)
print(v)

# Print the explained variance ratio of the PCA
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum()*100)

# Create a pipeline to scale the data and perform PCA
df = np.random.randn(20, 40)
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
pipeline.fit_transform(df)

# Detect and remove outliers from the dataset using a z-score threshold of 2.5
z_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
outlier_entries = (abs_z_scores >= 2.5).any(axis=1)
outliers = df[outlier_entries]
filtered_entries = (abs_z_scores < 2).all(axis=1)
df_filtered = df[filtered_entries]
print("Removed outliers:")
print(np.where(outlier_entries)[0])

# Read data from excel file and drop the first column
df = pd.read_excel("BClearning.xlsx")
df = df.iloc[:,1:]

# Calculate z-scores for numerical columns and filter out outliers
z_scores = np.abs(zscore(df.select_dtypes(include=['int', 'float'])))
outlier_entries = (z_scores >= 2.5).any(axis=1)
df_filtered = df.loc[~outlier_entries]

# Separate the features and target variable
x = df_filtered.drop(columns=["B.C."])
y = df_filtered["B.C."]

# Standardize the feature values
x_scaled = StandardScaler().fit_transform(x)

# Apply PCA to reduce dimensionality to 3
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])

# Combine the reduced features and target variable into a single dataframe
df_final = pd.concat([principalDf, y], axis = 1)

# Plot the 3D scatter plot
fig = px.scatter_3d(df_final, x='PC1', y='PC2', z='PC3', color='B.C.')
fig.show()

# Print the explained variance ratio and total percent variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total precent variance:", pca.explained_variance_ratio_.sum()*100)


# Read data from excel file and separate the features and target variable
df = pd.read_excel('BClearning.xlsx')
y = df['B.C.'].values
X = df.drop(columns=['B.C.']).values

# Standardize the feature values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply LDA to reduce dimensionality to 3
lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit_transform(X, y)

# Combine the reduced features and target variable into a single dataframe
lda_df = pd.DataFrame(data=X_lda, columns=['LD1', 'LD2', 'LD3'])
lda_df['B.C.'] = y

# Plot the 3D scatter plot
fig = px.scatter_3d(lda_df, x='LD1', y='LD2', z='LD3', color='B.C.')
fig.show()