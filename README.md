The code uses various data preprocessing techniques to reduce the dimensionality of the given dataset and visualize it in a 3D scatter plot. It also applies Linear Discriminant Analysis (LDA) to reduce dimensionality to 3 and visualize the dataset.

Firstly, the necessary libraries are imported and the dataset is loaded from an Excel file using Pandas. The 'id' column is dropped from the dataset and the target variable ('B.C.') is separated from the feature variables.

Then, the numeric columns are scaled to have a z-score of 0 and the covariance matrix for the z-scored data is computed. Principal Component Analysis (PCA) is then performed to reduce the dimensionality of the data to 3 components, and a new dataframe with the principal components and the target variable is created. Finally, a 3D scatter plot is created to visualize the data.

Next, the mean of the z-scored data is computed and the eigenvalues and eigenvectors of the covariance matrix are calculated. The explained variance ratio of the PCA is then printed.

A pipeline is created to scale the data and perform PCA. Outliers from the dataset are detected and removed using a z-score threshold of 2.5.

Finally, LDA is applied to reduce the dimensionality of the dataset to 3 and visualize the dataset in a 3D scatter plot. The explained variance ratio and total percent variance are printed.



<!DOCTYPE html>
<html>

<head>
  <meta charset="UTF-8">

</head>

<body>

  <h1>Breast Cancer Learning Project</h1>
  <p>This project uses machine learning techniques to analyze data related to breast cancer. The project is divided into several parts:</p>
  <ol>
    <li>Data preprocessing</li>
    <li>Dimensionality reduction using PCA (Principal Component Analysis)</li>
    <li>Outlier detection and removal</li>
    <li>Dimensionality reduction using LDA (Linear Discriminant Analysis)</li>
  </ol>

  <h2>Getting Started</h2>
  <p>To use this project, you will need to have Python installed on your computer. Additionally, you will need to install the following Python packages:</p>
  <ul>
    <li>pandas</li>
    <li>numpy</li>
    <li>scipy</li>
    <li>scikit-learn</li>
    <li>plotly</li>
    <li>matplotlib</li>
  </ul>
  <p>You can install these packages using pip. For example, to install pandas, you can run the following command:</p>
  <code>pip install pandas</code>

  <h2>Usage</h2>
  <p>To run the project, you can execute the Python code in the main.py file. The code is divided into several sections, each of which corresponds to one of the parts of the project.</p>
  <p>To run a specific part of the project, you can comment out the code for the other parts. For example, if you only want to run the PCA section of the project, you can comment out the code for the other sections.</p>

  <h2>Data</h2>
  <p>The data used in this project is stored in an Excel file named BClearning.xlsx. The file contains information about breast cancer patients, including various measurements and a binary classification variable (B.C.).</p>

  <h2>Results</h2>
  <p>The output of the project includes several plots that visualize the data after various transformations have been applied. These plots include:</p>
  <ul>
    <li>A 3D scatter plot of the data after PCA has been applied</li>
    <li>A 3D scatter plot of the data after LDA has been applied</li>
  </ul>
  <p>The output also includes information about the explained variance ratio of the PCA and the total percent variance.</p>

