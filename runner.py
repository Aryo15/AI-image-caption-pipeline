import torch
import numpy as np

# Upload and process image
import requests
from PIL import Image
from io import BytesIO

# Replace with your desired image URL
url = "https://images.pexels.com/photos/1103970/pexels-photo-1103970.jpeg"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")
image

# Detect objects
image, detections = detect_objects(image_path)

# Build graph
graph = build_graph(detections)

# Prepare features and adjacency matrix
node_feats = torch.tensor([[d["conf"], d["class"]] for d in detections], dtype=torch.float32)
adj_matrix = nx.to_numpy_array(graph)
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

# GNN Encoding
gnn_model = GCNEncoder(input_dim=2, hidden_dim=16, output_dim=8)
gnn_out = gnn_model(node_feats, adj_matrix)

# Generate caption
llm_input = gnn_to_llm_input(gnn_out)
caption = generate_caption(llm_input)

# Show output
from IPython.display import display
print("Caption:", caption)
display(image)
