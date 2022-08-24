from unittest import result
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import pickle as pkl

# result_PATH = "/home/wooyeon/MSIR/results/Glove_mosi.pkl"
# hidden_PATH = "/home/wooyeon/MSIR/hidden_vectors/Glove_mosi.pt"
# type_NAME = 'Glove'
result_PATH='/home/wooyeon/MSIR/results/COVAREP_mosi.pkl'
hidden_PATH='/home/wooyeon/MSIR/hidden_vectors/COVAREP_mosi.pt'
type_NAME = 'COVAREP'

# load test result dictionary ( segment / labels / labels_7 / labels_2 / preds / preds_7 / preds_2 )
with open(result_PATH, 'rb') as f:
    result_test = pkl.load(f)
testNUM = len(result_test['segment'])
# print dimension
print(testNUM)

# load hidden vector
embedding_matrix = torch.load(hidden_PATH)

# convert gpu tensor to array
for i in range(len(embedding_matrix)) :
    embedding_matrix[i] = embedding_matrix[i].cpu().numpy()

# shape of hidden vector    
print(np.shape(embedding_matrix))

# T-SNE
embedding_matrix = np.asarray(embedding_matrix)
tsne = TSNE(n_components=2).fit_transform(embedding_matrix)

# gold value
labels_7 = []
# predict value
preds_7 = []
# low dimension tensor
tensor = []
for i in range (0, testNUM) :
    labels_7.append(result_test['labels_7'][i])
    preds_7.append(result_test['preds_7'][i])
    tensor.append(tsne[i])
    
# Data Frame
tsne_df_7 = pd.DataFrame({'x': tsne[:, 0], 'y': tsne[:, 1], 'label': labels_7[:]})
print(tsne_df_7)

# visualization 7 class
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x = 'x', y = 'y',
    hue = 'label',
    hue_order = ['very positive', 'positive', 'weakly positive', 'Neutral', 'weakly negative', 'negative', 'very negative'],
    palette = ['darkgreen','green','mediumseagreen','burlywood','salmon','red','darkred'],
    data = tsne_df_7,
    legend = "full",
    alpha = 1,
)
plt.title("tSNE")
plt.savefig('/home/wooyeon/MSIR/src/'+type_NAME+'_7class.png', bbox_inches='tight')  
plt.show()
