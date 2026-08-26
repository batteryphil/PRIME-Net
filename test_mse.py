import torch
device = 'cpu'
X = torch.randn(1000)
X0 = X[5:999]
X1 = X[4:998]
Y_C = X0 * X1
Y_noisy = Y_C + torch.randn_like(Y_C) * 0.05
y_pred = X0 - X0
mse = torch.mean((y_pred - Y_noisy)**2).item()
print(f"MSE of (X0 - X0) on System C: {mse:.4f}")
