import networkx as nx
import pandas as pd
from karateclub import DeepWalk, Node2Vec
import pickle


edge_list = pd.read_csv("data/patent_edges.csv")
#edge_list = edge_list.applymap(str)
G=nx.from_pandas_edgelist(edge_list, source='node_1', target='node_2', create_using=nx.Graph())
print(len(G))

print("Fitting Deepwalk model...")
model = DeepWalk(walk_length=20, dimensions=32, window_size=4)
model.fit(G)
embeddings = model.get_embedding()

# print Embedding shape
print(embeddings.shape)

with open("patent_embeddings.pkl", 'wb') as pkl_file:
    pickle.dump(embeddings, pkl_file)