import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Step 1: Define your concepts
concepts = [
    "Machine Learning",
    "Neural Networks",
    "Supervised Learning",
    "Unsupervised Learning",
    "Deep Learning",
    "Reinforcement Learning"
]

# Step 2: Generate embeddings for each concept
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight model
embeddings = model.encode(concepts)

# Step 3: Compute similarity between concepts
similarity_matrix = cosine_similarity(embeddings)

# Step 4: Build concept graph
G = nx.Graph()
for i, concept in enumerate(concepts):
    G.add_node(concept)
    for j in range(i + 1, len(concepts)):
        similarity = similarity_matrix[i][j]
        if similarity > 0.5:  # link if similarity > threshold
            G.add_edge(concept, concepts[j], weight=similarity)

# Step 5: Visualize the concept map
pos = nx.spring_layout(G, seed=42)  # positions for nodes
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500, font_size=12)
edges = G.edges(data=True)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight']*2 for (u,v,d) in edges])
plt.title("MindMapr: Concept Linking")
plt.show()