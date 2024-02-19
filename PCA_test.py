import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from matplotlib.patches import Ellipse

Tk().withdraw()
fileName = askopenfilename()
df = pd.read_csv(fileName, header=None, sep=";")
df.dropna(inplace=True)
labels = df[0]
df = df.drop(labels=0, axis=1)

pca = PCA(n_components=2)
number_of_reps = 3

X_pca = pd.DataFrame(pca.fit(df).transform(df), columns = ["PC1", "PC2"])
X_pca["labels"] = labels

unique_labels = X_pca["labels"].unique()

plt.figure()
colors = ["navy", "turquoise", "purple", "darkorange", "pink"]
lw = 2

matplotlib.rcParams.update({'font.size': 16})

for name, color in zip(unique_labels, colors):
        category_data = X_pca[X_pca["labels"] == name]
        plt.scatter(
                category_data["PC1"], category_data["PC2"], s=80, alpha=0.8, lw=lw, label=name, color=color
                )
            
# Plot 95% confidence ellipses
for label, color in zip(unique_labels, colors):
    group_data = X_pca[X_pca["labels"] == label]
    cov_matrix = np.cov(group_data[["PC1", "PC2"]].T)
    lambda_, v = np.linalg.eig(cov_matrix)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(group_data["PC1"]), np.mean(group_data["PC2"])),
                  width=lambda_[0] * 2 * 2, height=lambda_[1] * 2 * 2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), alpha=0.5, color=color, lw=lw)
    ell.set_facecolor('none')
    plt.gca().add_patch(ell)
    # Annotate ellipse with label
    plt.annotate(label, xy=(np.mean(group_data["PC1"]), np.mean(group_data["PC2"])), xytext=(-10, 10),
                 textcoords='offset points', ha='right', va='top', fontsize=14, color='black')


plt.xlabel(f"F1 ({round(pca.explained_variance_ratio_[0], 4) * 100:.2f} %)")
plt.ylabel(f"F2 ({round(pca.explained_variance_ratio_[1], 4) * 100:.2f} %)")

plt.show()