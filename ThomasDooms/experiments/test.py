import numpy as np
import torch

nd = np.load('embeddings.npy')
print(nd.shape)

print(nd[1, :])
print(nd[2955044 // 2 + 1, :])

# for vec in nd[-1000:, :].tolist():
#     print(vec)

# print(nd.sum(axis=1))

# nd = torch.load('edges.pt')
# print(nd)
