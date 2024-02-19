import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

Tk().withdraw()
fileName = askopenfilename()
df = pd.read_csv(fileName, header=None, sep=";")
df.dropna(inplace=True)
labels = df[0]
df = df.drop(labels=0, axis=1)

lda = LDA(n_components=2)

number_of_samples = df.count()[1]

samples_correct = []

for i in range(number_of_samples):
    sample = df.iloc[[i]]

    df_new = df.drop(i)
    labels_new = labels.drop(i)

    X_lda = lda.fit_transform(df_new, y=labels_new)
   
    # Compute within-class scatter matrix
    classes = np.unique(labels_new)
    within_class_scatter_matrix = np.zeros((X_lda.shape[1], X_lda.shape[1]))
    for cls in classes:
        X_cls = X_lda[labels_new == cls]
        scatter_matrix = np.cov(X_cls, rowvar=False)
        within_class_scatter_matrix += (len(X_cls) / len(df_new)) * scatter_matrix

    inv_within_class_scatter_matrix = np.linalg.inv(within_class_scatter_matrix)
    critical_value = chi2.ppf(0.95, df=2)

    # Calculate Mahalanobis distance for the sample in the transformed space
    sample_transformed = lda.transform(sample)[0]  # Extract 1-D array
    mean_transformed = np.mean(X_lda, axis=0)
    mahalanobis_distance = mahalanobis(sample_transformed, mean_transformed, inv_within_class_scatter_matrix)
    if mahalanobis_distance <= critical_value:
        samples_correct.append(i)

print("Samples within the 95% confidence interval:", samples_correct)