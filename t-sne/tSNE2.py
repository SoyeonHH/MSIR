from unittest import result
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import pickle as pkl

# load label
with open("/home/wooyeon/MSIR/results/COVAREP_mosi.pkl", 'rb') as f:
    result_test = pkl.load(f)
    
print('segment', result_test['segment'][0])
print('labels', result_test['labels'][0])
print('labels_2', result_test['labels_2'][0])
print('labels_7', result_test['labels_7'][0])
print('preds', result_test['preds'][0])
print('preds_2', result_test['preds_2'][0])
print('preds_7', result_test['preds_7'][0])
    


# load hidden vector
embedding_matrix = torch.load("/home/wooyeon/MSIR/hidden_vectors/COVAREP_mosi.pt")

# convert gpu tensor to array
for i in range(len(embedding_matrix)) :
    embedding_matrix[i] = embedding_matrix[i].cpu().numpy()

# print(embedding_matrix[1])
# shape of hidden vector    
print(np.shape(embedding_matrix))

# T-SNE
embedding_matrix = np.asarray(embedding_matrix)
tsne = TSNE(n_components=2).fit_transform(embedding_matrix)

# add 'tensor' dictionary 
result_test['tensor'] = tsne

# print(np.shape(tsne))
# print(tsne[0])
# print(np.shape(result_test['tensor']))
# print(result_test['tensor'][0])
labels_7_dict = []
labels_2_dict = []
preds_7_dict = []
preds_2_dict = []
tensor = []
for i in range (0,686) :
    labels_7_dict.append(result_test['labels_7'][i])
    labels_2_dict.append(result_test['labels_2'][i])
    preds_7_dict.append(result_test['preds_7'][i])
    preds_2_dict.append(result_test['preds_2'][i])
    tensor.append(tsne[i])
    

tsne_df = pd.DataFrame({'x': tsne[:, 0], 'y':tsne[:, 1], 'label':labels_7_dict[:]})
print(tsne_df)

# visualization
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x = 'x', y = 'y',
    hue = 'label',
    # palette = sns.color_palette("Set1", 10),
    palette = sns.color_palette('deep',7),
    data = tsne_df,
    legend = "full",
    alpha = 0.4
)

plt.title("tSNE")

plt.savefig('TSNE.png', bbox_inches='tight')  
plt.show()


