import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        h = self.linear(h)
        return F.relu(h)

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x


from transformers import AutoTokenizer, AutoModelForCausalLM

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# Inject GNN output as token embeddings
def gnn_to_llm_input(gnn_output):
    flat = gnn_output.mean(dim=0)  # mean pooling
    fake_text = "gnn: " + " ".join([f"{x:.2f}" for x in flat.tolist()])
    return fake_text

def generate_caption(gnn_embedding_text, max_length=50):
    prompt = f"Image graph info: {gnn_embedding_text}. Caption:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = llm.generate(**inputs, max_length=max_length, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
