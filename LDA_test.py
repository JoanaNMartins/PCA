import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

lda = LDA(n_components=2)

X_lda = pd.DataFrame(lda.fit_transform(df, y=labels), columns = ["LD1", "LD2"])
X_lda["labels"] = labels

unique_labels = X_lda["labels"].unique()

plt.figure()
colors = ["navy", "turquoise", "purple", "darkorange", "pink"]
lw = 2

matplotlib.rcParams.update({'font.size': 16})

for name, color in zip(unique_labels, colors):
        category_data = X_lda[X_lda["labels"] == name]
        plt.scatter(
                category_data["LD1"], category_data["LD2"], s=80, alpha=0.8, lw=lw, label=name, color=color
                )

# Plot 95% confidence ellipses
for label, color in zip(unique_labels, colors):
    group_data = X_lda[X_lda["labels"] == label]
    cov_matrix = np.cov(group_data[["LD1", "LD2"]].T)
    lambda_, v = np.linalg.eig(cov_matrix)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(group_data["LD1"]), np.mean(group_data["LD2"])),
                  width=lambda_[0] * 2 * 2, height=lambda_[1] * 2 * 2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), alpha=0.2, color=color)
    ell.set_facecolor('none')
    plt.gca().add_patch(ell)



plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.xlabel(f"F1 ({round(lda.explained_variance_ratio_[0], 4) * 100:.2f} %)")
plt.ylabel(f"F2 ({round(lda.explained_variance_ratio_[1], 4) * 100:.2f} %)")

plt.show()