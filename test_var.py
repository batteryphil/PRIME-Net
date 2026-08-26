import torch
device = 'cpu'
X = torch.randn(1000)
X0 = X[5:999]
X1 = X[4:998]
X2 = X[3:997]
Y_A = (X0 + X2) / 2.0
Y_B = torch.sin(X0) + X1
Y_C = X0 * X1
Y_D = torch.abs(X0 - X1)
print(f"Var A: {Y_A.var().item():.4f}")
print(f"Var B: {Y_B.var().item():.4f}")
print(f"Var C: {Y_C.var().item():.4f}")
print(f"Var D: {Y_D.var().item():.4f}")
