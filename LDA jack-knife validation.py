import matplotlib.pyplot as plt
import matplotlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd

Tk().withdraw()
fileName = askopenfilename()
df = pd.read_csv(fileName, header=None, sep=";")
df.dropna(inplace=True)
labels = df[0]
df = df.drop(labels=0, axis=1)

sampleName = askopenfilename()
sample = pd.read_csv(sampleName, header=None, sep=";")
sample.dropna(inplace=True)

lda = LDA(n_components=2)

X_lda = pd.DataFrame(lda.fit_transform(df, y=labels), columns = ["LD1", "LD2"])
X_lda["labels"] = labels

X_sample = lda.transform(sample)

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

print(X_sample)

plt.scatter(X_sample[0][0], X_sample[0][1], marker="x", color = "black", label="Test Sample", s=80)

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.xlabel(f"F1 ({round(lda.explained_variance_ratio_[0], 4) * 100:.2f} %)")
plt.ylabel(f"F2 ({round(lda.explained_variance_ratio_[1], 4) * 100:.2f} %)")

plt.show()