#math
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#debugging
import os #listdir

#make sure we have the right path (change line below if not)
csv_path="../../Embeddings/Gemini_sum_embeddings-clean.csv"
# print(os.listdir(csv_path))

#read CSV
df=pd.read_csv(csv_path)

# Show the first few rows
print(df.head())

# Convert embeddings of the format "[0.1, 0.2, ..., 0.5]" to real lists
if isinstance(df.iloc[0]['EMBEDDING'], str):
    df['EMBEDDING'] = df['EMBEDDING'].apply(lambda x: np.array(eval(x)))

# Stack embeddings into a 2D array
embeddings = np.stack(df['EMBEDDING'].values)

#------------------ 2D Visualization ------------------#
pca_2d = PCA(n_components=2)
embeddings_2d = pca_2d.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
plt.title('2D Visualization of Gemini Summarization Embeddings (PCA)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

#------------------ 3D Visualization ------------------#
pca_3d = PCA(n_components=3)
embeddings_3d = pca_3d.fit_transform(embeddings)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], alpha=0.6)

ax.set_title('3D Visualization of Gemini Summarization Embeddings (PCA)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()
