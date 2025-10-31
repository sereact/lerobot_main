import torch

data = torch.load("/home/ubuntu/mount-point/libero/rollout_20251023_115222.pt", map_location="cpu")

print(data.keys())
# dict_keys(['actions', 'rewards', 'successes', 'dones', 'observations'])

print(data["actions"].shape)
print(data["observations"].keys())
