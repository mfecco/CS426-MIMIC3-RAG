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
csv_path="../../Embeddings/Bart_sum_embeddings.csv"
# print(os.listdir(csv_path))

#read CSV
df=pd.read_csv(csv_path)

#show the first few rows
print(df.head())

#convert embeddings of the format "[0.1,0.2,...,0.5]" to lists
if isinstance(df.iloc[0]['EMBEDDING'],str):
    df['EMBEDDING']=df['EMBEDDING'].apply(lambda x: np.array(eval(x)))

#convert embeddings to a reduced 2D-array
embeddings=np.stack(df['EMBEDDING'].values)


#----------- 2D Vis ---------------- 

#Reduce dimensions (Deciding between PCA for speed and TSNE for prettier separation)
pca_2d=PCA(n_components=2)
embeddings_2d=pca_2d.fit_transform(embeddings)

# tsne=TSNE(n_components=2,random_state=42)
# embeddings_2d=tsne.fit_transform(embeddings)

#matplotlib
plt.figure(figsize=(10,7))
plt.scatter(embeddings_2d[:,0],embeddings_2d[:,1],alpha=0.6)
plt.title('2D Visualization of Summarization Embeddings')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

#----------- 3D Vis ---------------- 
pca_3d=PCA(n_components=3)
embeddings_3d=pca_3d.fit_transform(embeddings)

fig=plt.figure(figsize=(12,9))
ax=fig.add_subplot(111,projection='3d')
sc=ax.scatter(embeddings_3d[:,0],embeddings_3d[:,1],embeddings_3d[:,2],alpha=0.6)

ax.set_title('3D Visualization of BART Summarization Embeddings (PCA)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
plt.show()