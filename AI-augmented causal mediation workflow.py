#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
# DoWhy
import dowhy
from dowhy import CausalModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


# ==========================
# 1. 模拟数据
# ==========================
np.random.seed(42)
torch.manual_seed(42)

n_samples = 500
n_features = 3   # X: age, sex, bmi
n_mediators = 2  # M
n_diseases = 3   # Y: disease network节点数
n_genes = 5      # G: 基因特征数

# Treatment (binary)
T = np.random.binomial(1, 0.5, size=(n_samples, 1))

# Covariates X
X = np.random.randn(n_samples, n_features)

# Mediators M
M = np.random.randn(n_samples, n_mediators)

# Genomic data G
G = np.random.randn(n_samples, n_genes)

# Outcome: disease network, 多维连续
Y = np.random.randn(n_samples, n_diseases)

# 转成 tensor
T_tensor = torch.tensor(T, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)
M_tensor = torch.tensor(M, dtype=torch.float32)
G_tensor = torch.tensor(G, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# ==========================
# 2. cVAE 编码 M
# ==========================
class CVAE_M(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + cond_dim, input_dim)

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.fc_decode(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar, z

latent_dim = 2
cond_dim = n_features + 1 + n_genes  # X + T + G
cvae_M = CVAE_M(input_dim=n_mediators, cond_dim=cond_dim, latent_dim=latent_dim)
optimizer = optim.Adam(cvae_M.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# Training cVAE
n_epochs = 200
for epoch in range(n_epochs):
    optimizer.zero_grad()
    x_hat, mu, logvar, z = cvae_M(M_tensor, torch.cat([T_tensor, X_tensor, G_tensor], dim=1))
    recon_loss = loss_fn(x_hat, M_tensor)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.2f}")

# 获取编码后的中介
with torch.no_grad():
    _, _, _, Z_M = cvae_M(M_tensor, torch.cat([T_tensor, X_tensor, G_tensor], dim=1))
Z_M = Z_M.numpy()  # shape: [n_samples, latent_dim]

# ==========================
# 3. 用Dowhy做因果分析
# 针对每个 disease 节点单独计算 ATE
# ==========================
feature_cols = ['age','sex','bmi'] + [f'G{j+1}' for j in range(n_genes)]

for i in range(n_diseases):
    data = pd.DataFrame(
        np.hstack([T, Z_M, X, G, Y[:, i:i+1]]),
        columns=['T'] + [f'Z_M{j+1}' for j in range(latent_dim)] + feature_cols + ['Y']
    )

    model = CausalModel(
        data=data,
        treatment='T',
        outcome='Y',
        common_causes=feature_cols + [f'Z_M{j+1}' for j in range(latent_dim)]
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression")
    print(f"Disease node {i+1} ATE:", estimate.value)


# # Version 2: with fine-tuning

# In[5]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dowhy import CausalModel

# ==========================
# 1. 配置与超参数
# ==========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

n_samples = 500
n_features = 3    # X: age, sex, bmi
n_mediators = 2   # M
n_diseases = 3    # Y
n_genes = 5       # G

latent_dim = 2
cvae_hidden = 16
learning_rate = 1e-2
n_epochs = 200
print_interval = 50

# ==========================
# 2. 模拟数据生成
# ==========================
T = np.random.binomial(1, 0.5, size=(n_samples, 1))
X = np.random.randn(n_samples, n_features)
M = np.random.randn(n_samples, n_mediators)
G = np.random.randn(n_samples, n_genes)
Y = np.random.randn(n_samples, n_diseases)

# 转为 tensor
T_tensor = torch.tensor(T, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)
M_tensor = torch.tensor(M, dtype=torch.float32)
G_tensor = torch.tensor(G, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# ==========================
# 3. cVAE 定义 (M)
# ==========================
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=2, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + cond_dim, input_dim)

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.fc_decode(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar, z

# ==========================
# 4. 初始化 cVAE
# ==========================
cond_dim = n_features + 1 + n_genes  # X + T + G
cvae_M = CVAE(input_dim=n_mediators, cond_dim=cond_dim, latent_dim=latent_dim, hidden_dim=cvae_hidden)
optimizer = optim.Adam(cvae_M.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# ==========================
# 5. 训练 cVAE
# ==========================
for epoch in range(n_epochs):
    optimizer.zero_grad()
    x_hat, mu, logvar, z = cvae_M(M_tensor, torch.cat([T_tensor, X_tensor, G_tensor], dim=1))
    recon_loss = loss_fn(x_hat, M_tensor)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    loss.backward()
    optimizer.step()
    if (epoch+1) % print_interval == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

with torch.no_grad():
    _, _, _, Z_M = cvae_M(M_tensor, torch.cat([T_tensor, X_tensor, G_tensor], dim=1))
Z_M = Z_M.numpy()  # latent representation of M

# ==========================
# 6. 因果分析 (DoWhy) - 每个 disease 节点单独 ATE
# ==========================
feature_cols = ['age', 'sex', 'bmi'] + [f'G{j+1}' for j in range(n_genes)]

for i in range(n_diseases):
    data = pd.DataFrame(
        np.hstack([T, Z_M, X, G, Y[:, i:i+1]]),
        columns=['T'] + [f'Z_M{j+1}' for j in range(latent_dim)] + feature_cols + ['Y']
    )

    model = CausalModel(
        data=data,
        treatment='T',
        outcome='Y',
        common_causes=feature_cols + [f'Z_M{j+1}' for j in range(latent_dim)]
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print(f"Disease node {i+1} ATE: {estimate.value:.4f}")


# #  Version 3
# 
# We use a joint cVAE setup for causal mediation analysis. Basically, **cVAE_M** compresses the psychological mediators (M) while taking into account treatment (T), covariates (X), and genomic data (G), and **cVAE_Y** compresses the health outcomes (Y) using (T, X, G) plus the latent mediators (Z_M). This gives us compact representations (Z_M) and (Z_Y) that can capture the tricky, nonlinear relationships from (M) to (Y). Then, the full pipeline brings together (T, X, G, Z_M, Z_Y) for downstream mediation analysis, so we can estimate natural direct and indirect effects while also accounting for genetic factors.

# In[6]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dowhy import CausalModel

# ==========================
# 1. 配置与超参数
# ==========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

n_samples = 500
n_features = 3     # X: age, sex, bmi
n_mediators = 2    # M
n_diseases = 3     # Y
n_genes = 5        # G

latent_dim = 2
hidden_dim = 16
learning_rate = 1e-2
n_epochs = 300
print_interval = 50

# ==========================
# 2. 模拟数据
# ==========================
T = np.random.binomial(1, 0.5, size=(n_samples, 1))
X = np.random.randn(n_samples, n_features)
M = np.random.randn(n_samples, n_mediators)
G = np.random.randn(n_samples, n_genes)
Y = np.random.randn(n_samples, n_diseases)

# 转为 tensor
T_tensor = torch.tensor(T, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)
M_tensor = torch.tensor(M, dtype=torch.float32)
G_tensor = torch.tensor(G, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# ==========================
# 3. 联合 cVAE 定义 (M + Y)
# ==========================
class JointCVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=2, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + cond_dim, input_dim)

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.fc_decode(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar, z

# 条件维度 = X + T + G
cond_dim_M = n_features + 1 + n_genes
cond_dim_Y = n_features + 1 + n_genes + latent_dim  # Y 条件包括 Z_M

cvae_M = JointCVAE(input_dim=n_mediators, cond_dim=cond_dim_M, latent_dim=latent_dim, hidden_dim=hidden_dim)
cvae_Y = JointCVAE(input_dim=n_diseases, cond_dim=cond_dim_Y, latent_dim=latent_dim, hidden_dim=hidden_dim)

optimizer = optim.Adam(list(cvae_M.parameters()) + list(cvae_Y.parameters()), lr=learning_rate)
loss_fn = nn.MSELoss()

# ==========================
# 4. 联合训练 cVAE
# ==========================
for epoch in range(n_epochs):
    optimizer.zero_grad()

    # M 编码
    x_hat_M, mu_M, logvar_M, Z_M = cvae_M(M_tensor, torch.cat([T_tensor, X_tensor, G_tensor], dim=1))
    recon_loss_M = loss_fn(x_hat_M, M_tensor)
    kl_loss_M = -0.5 * torch.mean(1 + logvar_M - mu_M.pow(2) - logvar_M.exp())

    # Y 编码，条件包括 Z_M
    x_hat_Y, mu_Y, logvar_Y, Z_Y = cvae_Y(Y_tensor, torch.cat([T_tensor, X_tensor, G_tensor, Z_M], dim=1))
    recon_loss_Y = loss_fn(x_hat_Y, Y_tensor)
    kl_loss_Y = -0.5 * torch.mean(1 + logvar_Y - mu_Y.pow(2) - logvar_Y.exp())

    # 总 loss
    loss = recon_loss_M + kl_loss_M + recon_loss_Y + kl_loss_Y
    loss.backward()
    optimizer.step()

    if (epoch+1) % print_interval == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# ==========================
# 5. DoWhy 因果分析 (ATE)
# ==========================
Z_M = Z_M.detach().numpy()
Z_Y = Z_Y.detach().numpy()

feature_cols = ['age', 'sex', 'bmi'] + [f'G{j+1}' for j in range(n_genes)]

for i in range(n_diseases):
    data = pd.DataFrame(
        np.hstack([T, Z_M, X, G, Y[:, i:i+1]]),
        columns=['T'] + [f'Z_M{j+1}' for j in range(latent_dim)] + feature_cols + ['Y']
    )

    model = CausalModel(
        data=data,
        treatment='T',
        outcome='Y',
        common_causes=feature_cols + [f'Z_M{j+1}' for j in range(latent_dim)]
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print(f"Disease node {i+1} ATE: {estimate.value:.4f}")


# In[ ]:




