import os.path

import torch
import time
import pandas as pd
import numpy as np
from torch_geometric.nn import Node2Vec

from ThomasDooms.src.paths import path

articles = pd.read_feather(path('articles', 'full'))
customers = pd.read_feather(path('customers', 'full'))

if not os.path.exists('edges.pt'):
    start = time.time()
    transactions = pd.read_feather(path('transactions', 'full'))

    a_map = {a_id: idx for idx, a_id in enumerate(articles['article_id'].values)}
    c_map = {c_id: idx + len(articles) for idx, c_id in enumerate(customers['customer_id'].values)}

    transactions["article_id"] = transactions["article_id"].map(a_map)
    transactions["customer_id"] = transactions["customer_id"].map(c_map)

    transactions['article_id'] = transactions['article_id'].astype(np.int32)
    transactions['customer_id'] = transactions['customer_id'].astype(np.int32)

    # convert to torch tensors
    edge_index1 = torch.tensor(transactions[['article_id', 'customer_id']].values.T, dtype=torch.long)
    edge_index2 = torch.tensor(transactions[['customer_id', 'article_id']].values.T, dtype=torch.long)

    # concatenate the two edge indices and save it for future use
    edge_index = torch.cat([edge_index1, edge_index2], dim=1)
    torch.save(edge_index, 'edges.pt')

    print(f"done generating data in {time.time() - start:.2f} seconds")
else:
    start = time.time()
    edge_index = torch.load('edges.pt')
    print(f"done reading data in {time.time() - start:.2f} seconds")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

size = len(articles) + len(customers)
model = Node2Vec(edge_index,
                 embedding_dim=16,
                 walk_length=10,
                 context_size=10,
                 walks_per_node=5,
                 num_negative_samples=1,
                 p=1,
                 q=1,
                 num_nodes=size,
                 sparse=True).to(device)

loader = model.loader(batch_size=256, shuffle=True, num_workers=0)  # data loader to speed the train
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)  # initialise the optimizer


def train():
    model.train()  # put model in train model
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()  # set the gradients to 0
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # compute the loss for the batch
        loss.backward()
        optimizer.step()  # optimize the parameters
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    start = time.time()
    for epoch in range(1, 30):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f} after {time.time() - start} seconds')

    nodes = torch.arange(size, device=device)

    embeddings = [tensor.detach().cpu().numpy() for tensor in model(nodes)]
    np.save('embeddings3.npy', np.array(embeddings))


if __name__ == "__main__":
    main()
