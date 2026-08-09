#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics.pairwise import euclidean_distances

from numpy.linalg import cond

 

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print(device)

 

nx = 50
ny = 50

N = nx * ny

dx = 1
dy = 1

 

D = 0.03

v_x = 4.0

dt = 0.01

nt = 100

 

def idx(i,j):

    return i * ny + j

 

u = np.zeros((nx,ny))

u[20:25,10:15] = 10

 

fields_over_time = []

fields_over_time.append(
    u.copy()
)

 

T_true = np.zeros((N,N))

for i in range(1,nx-1):

    for j in range(1,ny-1):

        center = idx(i,j)

       

        self_weight = (

            1

            -

            v_x * dt / dx

            -

            4 * D * dt
        )

        adv_weight = (

            v_x * dt / dx

            +

            D * dt
        )

        diff_weight = D * dt

        

        T_true[center,center] = self_weight

        # transport direction

        T_true[center,idx(i-1,j)] = adv_weight

        # diffusion

        T_true[center,idx(i+1,j)] = diff_weight

        T_true[center,idx(i,j+1)] = diff_weight

        T_true[center,idx(i,j-1)] = diff_weight

 

for t in range(nt):

    u_vec = u.reshape(-1)

    u_next = np.matmul(

        T_true,

        u_vec
    )

    u = u_next.reshape(nx,ny)

    fields_over_time.append(
        u.copy()
    )

 

plt.figure(figsize=(7,6))

plt.imshow(
    u,
    origin="lower"
)

plt.title(
    "Final Transport Field"
)

plt.colorbar()

plt.show()

 

E_true = []

for t in range(len(fields_over_time)-1):

    current_field = fields_over_time[t]

    exposure = np.matmul(

        T_true,

        current_field.reshape(-1)
    )

    E_true.append(
        exposure
    )

E_true = np.array(E_true)

print(E_true.shape)

 

coords = []

for i in range(nx):

    for j in range(ny):

        coords.append([i,j])

coords = np.array(coords)

dist_matrix = euclidean_distances(coords)

sigma = 4

W_gaussian = np.exp(

    -(dist_matrix**2)/(2*sigma**2)
)

W_gaussian = (

    W_gaussian

    /

    W_gaussian.sum(
        axis=1,
        keepdims=True
    )
)

E_gaussian = []

for t in range(len(fields_over_time)-1):

    exposure = np.matmul(

        W_gaussian,

        fields_over_time[t].reshape(-1)
    )

    E_gaussian.append(
        exposure
    )

E_gaussian = np.array(
    E_gaussian
)

 

U_t = []

U_next = []

for t in range(len(fields_over_time)-1):

    current_field = fields_over_time[t]

    next_field = fields_over_time[t+1]

    U_t.append(

        current_field.reshape(-1)
    )

    U_next.append(

        next_field.reshape(-1)
    )

U_t = np.array(U_t)

U_next = np.array(U_next)

 

np.random.seed(123)

n_monitors = 50

monitor_idx = np.random.choice(

    N,

    size=n_monitors,

    replace=False
)

U_t_sparse = U_t[:,monitor_idx]

U_next_sparse = U_next[:,monitor_idx]

print(U_t_sparse.shape)

 

U_t_torch = torch.tensor(

    U_t,

    dtype=torch.float32
).to(device)

U_next_torch = torch.tensor(

    U_next,

    dtype=torch.float32
).to(device)

 

T_param = nn.Parameter(

    0.001 * torch.randn(N,N)
)

optimizer = optim.Adam(

    [T_param],

    lr=0.001
)

 

local_mask = np.zeros((N,N))

for i in range(nx):

    for j in range(ny):

        center = idx(i,j)

        neighbors = [

            (i,j),

            (i+1,j),

            (i-1,j),

            (i,j+1),

            (i,j-1)
        ]

        for ni,nj in neighbors:

            if (

                0 <= ni < nx

                and

                0 <= nj < ny
            ):

                local_mask[

                    center,

                    idx(ni,nj)

                ] = 1

local_mask = torch.tensor(

    local_mask,

    dtype=torch.float32
).to(device)

 

epochs = 3000

loss_history = []

for epoch in range(epochs):

    optimizer.zero_grad()

    

    T_local = (

        T_param

        *

        local_mask
    )

  

    T_positive = torch.relu(
        T_local
    )

  

    row_sums = (

        T_positive.sum(
            dim=1,
            keepdim=True
        )

        +

        1e-6
    )

    T_norm = (

        T_positive

        /

        row_sums
    )

 

    U_pred = torch.matmul(

        U_t_torch,

        T_norm.T
    )

 

    U_pred_sparse = U_pred[:,monitor_idx]

    U_next_sparse_torch = U_next_torch[:,monitor_idx]

    loss_temporal = torch.mean(

        (
            U_pred_sparse

            -

            U_next_sparse_torch
        )**2
    )

 

    loss_sparse = torch.mean(
        torch.abs(T_norm)
    )

 

    loss = (

        loss_temporal

        +

        0.0001 * loss_sparse
    )

    loss.backward()

    optimizer.step()

    loss_history.append(
        loss.item()
    )

    if epoch % 500 == 0:

        print(

            "Epoch:", epoch,

            "Loss:", round(loss.item(),6)
        )

 

T_learned = (

    T_norm

    .detach()

    .cpu()

    .numpy()
)

 

E_learned = []

for t in range(len(fields_over_time)-1):

    exposure = np.matmul(

        T_learned,

        fields_over_time[t].reshape(-1)
    )

    E_learned.append(
        exposure
    )

E_learned = np.array(
    E_learned
)

 
rmse_gaussian = np.sqrt(

    np.mean(

        (
            E_true

            -

            E_gaussian
        )**2
    )
)

rmse_learned = np.sqrt(

    np.mean(

        (
            E_true

            -

            E_learned
        )**2
    )
)

print()

print("Gaussian Mapping RMSE:")

print(round(rmse_gaussian,4))

print()

print("Learned Mapping RMSE:")

print(round(rmse_learned,4))

 

fro_error_gaussian = (

    np.linalg.norm(

        T_true

        -

        W_gaussian,

        ord="fro"
    )

    /

    np.linalg.norm(
        T_true,
        ord="fro"
    )
)

fro_error_learned = (

    np.linalg.norm(

        T_true

        -

        T_learned,

        ord="fro"
    )

    /

    np.linalg.norm(
        T_true,
        ord="fro"
    )
)

print()

print("Gaussian Operator Error:")

print(round(fro_error_gaussian,4))

print()

print("Learned Operator Error:")

print(round(fro_error_learned,4))

 

corr_gaussian = np.corrcoef(

    T_true.flatten(),

    W_gaussian.flatten()
)[0,1]

corr_learned = np.corrcoef(

    T_true.flatten(),

    T_learned.flatten()
)[0,1]

print()

print("Gaussian Correlation:")

print(round(corr_gaussian,4))

print()

print("Learned Correlation:")

print(round(corr_learned,4))

 

cond_gaussian = cond(

    W_gaussian + 1e-6*np.eye(N)
)

cond_learned = cond(

    T_learned + 1e-6*np.eye(N)
)

print()

print("Gaussian Condition Number:")

print(round(cond_gaussian,4))

print()

print("Learned Condition Number:")

print(round(cond_learned,4))


# In[ ]:


fast_results = []

 
monitor_grid = [

    100,
    50,
    20,
    10
]

sigma_grid = [

    2,
    4,
    8,
    16
]

 

for sigma in sigma_grid:

    print()

 

    print("SIGMA:", sigma)

 
 

    W_gaussian = np.exp(

        -(dist_matrix**2)/(2*sigma**2)
    )

    W_gaussian = (

        W_gaussian

        /

        W_gaussian.sum(
            axis=1,
            keepdims=True
        )
    )

 

    gaussian_error = (

        np.linalg.norm(

            T_true

            -

            W_gaussian,

            ord="fro"
        )

        /

        np.linalg.norm(
            T_true,
            ord="fro"
        )
    )

    gaussian_corr = np.corrcoef(

        T_true.flatten(),

        W_gaussian.flatten()
    )[0,1]

    gaussian_cond = cond(

        W_gaussian + 1e-6*np.eye(N)
    )

 

    for n_monitors in monitor_grid:

        print()

        print("Monitors:", n_monitors)

    

        np.random.seed(123)

        monitor_idx = np.random.choice(

            N,

            size=n_monitors,

            replace=False
        )

  

        theta_self = nn.Parameter(
            torch.tensor(0.7)
        )

        theta_adv = nn.Parameter(
            torch.tensor(0.2)
        )

        theta_diff = nn.Parameter(
            torch.tensor(0.05)
        )

        optimizer = optim.Adam(

            [
                theta_self,
                theta_adv,
                theta_diff
            ],

            lr=0.01
        )

     

        epochs = 500

        for epoch in range(epochs):

            optimizer.zero_grad()

      

            T_struct = torch.zeros(

                (N,N),

                dtype=torch.float32
            ).to(device)

            for i in range(1,nx-1):

                for j in range(1,ny-1):

                    center = idx(i,j)

                    T_struct[
                        center,
                        center
                    ] = torch.relu(theta_self)

                    T_struct[
                        center,
                        idx(i-1,j)
                    ] = torch.relu(theta_adv)

                    T_struct[
                        center,
                        idx(i+1,j)
                    ] = torch.relu(theta_diff)

                    T_struct[
                        center,
                        idx(i,j+1)
                    ] = torch.relu(theta_diff)

                    T_struct[
                        center,
                        idx(i,j-1)
                    ] = torch.relu(theta_diff)

 

            row_sums = (

                T_struct.sum(
                    dim=1,
                    keepdim=True
                )

                +

                1e-6
            )

            T_struct = (

                T_struct

                /

                row_sums
            )

   

            U_pred = torch.matmul(

                U_t_torch,

                T_struct.T
            )

            U_pred_sparse = U_pred[:,monitor_idx]

            U_next_sparse = U_next_torch[:,monitor_idx]

            

            loss = torch.mean(

                (
                    U_pred_sparse

                    -

                    U_next_sparse
                )**2
            )

            loss.backward()

            optimizer.step()

 

        T_final = (

            T_struct

            .detach()

            .cpu()

            .numpy()
        )

 

        operator_error = (

            np.linalg.norm(

                T_true

                -

                T_final,

                ord="fro"
            )

            /

            np.linalg.norm(
                T_true,
                ord="fro"
            )
        )

        operator_corr = np.corrcoef(

            T_true.flatten(),

            T_final.flatten()
        )[0,1]

        operator_cond = cond(

            T_final + 1e-6*np.eye(N)
        )

 

        fast_results.append({

            "sigma": sigma,

            "monitors": n_monitors,

            "gaussian_error":
                gaussian_error,

            "structured_error":
                operator_error,

            "gaussian_corr":
                gaussian_corr,

            "structured_corr":
                operator_corr,

            "gaussian_condition":
                gaussian_cond,

            "structured_condition":
                operator_cond,

            "theta_self":
                theta_self.item(),

            "theta_adv":
                theta_adv.item(),

            "theta_diff":
                theta_diff.item()
        })

 

fast_df = pd.DataFrame(
    fast_results
)

print()

print(fast_df.head())

 

fast_df.to_csv(

    "fast_operator_results.csv",

    index=False
)

print()

print("Saved:")
print("fast_operator_results.csv")


# In[ ]:


center_idx = []
left_idx = []
right_idx = []
up_idx = []
down_idx = []

for i in range(1,nx-1):

    for j in range(1,ny-1):

        center = idx(i,j)

        center_idx.append(center)

        left_idx.append(idx(i-1,j))

        right_idx.append(idx(i+1,j))

        up_idx.append(idx(i,j+1))

        down_idx.append(idx(i,j-1))

center_idx = torch.tensor(center_idx).to(device)

left_idx = torch.tensor(left_idx).to(device)

right_idx = torch.tensor(right_idx).to(device)

up_idx = torch.tensor(up_idx).to(device)

down_idx = torch.tensor(down_idx).to(device)

 
theta_self = nn.Parameter(
    torch.tensor(0.7).to(device)
)

theta_adv = nn.Parameter(
    torch.tensor(0.2).to(device)
)

theta_diff = nn.Parameter(
    torch.tensor(0.05).to(device)
)

optimizer = optim.Adam(

    [
        theta_self,
        theta_adv,
        theta_diff
    ],

    lr=0.01
)

 

epochs = 300

for epoch in range(epochs):

    optimizer.zero_grad()

 

    T_struct = torch.zeros(

        (N,N),

        device=device
    )

    T_struct[
        center_idx,
        center_idx
    ] = torch.relu(theta_self)

    T_struct[
        center_idx,
        left_idx
    ] = torch.relu(theta_adv)

    T_struct[
        center_idx,
        right_idx
    ] = torch.relu(theta_diff)

    T_struct[
        center_idx,
        up_idx
    ] = torch.relu(theta_diff)

    T_struct[
        center_idx,
        down_idx
    ] = torch.relu(theta_diff)

 

    row_sums = (

        T_struct.sum(
            dim=1,
            keepdim=True
        )

        +

        1e-6
    )

    T_struct = (

        T_struct

        /

        row_sums
    )

  

    U_pred = torch.matmul(

        U_t_torch,

        T_struct.T
    )

    U_pred_sparse = U_pred[:,monitor_idx]

    U_next_sparse = U_next_torch[:,monitor_idx]

    

    loss = torch.mean(

        (
            U_pred_sparse

            -

            U_next_sparse
        )**2
    )

    loss.backward()

    optimizer.step()

  

    if epoch % 50 == 0:

        print(

            "Epoch:",
            epoch,

            "Loss:",
            round(loss.item(),6)
        )

 

T_final = (

    T_struct

    .detach()

    .cpu()

    .numpy()
)


# In[ ]:


import pandas as pd
structured_results = []

 

adv_grid = [

    0.1,
    0.6
]

diff_grid = [

    0.01,
    0.10
]

monitor_grid = [

    100,
    10
]

sigma_grid = [

    2,
    16
]

 

coords = []

for i in range(nx):

    for j in range(ny):

        coords.append([i,j])

coords = np.array(coords)

dist_matrix = euclidean_distances(coords)

 

for theta_adv_true in adv_grid:

    for theta_diff_true in diff_grid:

        print()
 

        print("ADV:", theta_adv_true)

        print("DIFF:", theta_diff_true)

 

        T_true_struct = np.zeros((N,N))

        theta_self_true = (

            1

            -

            theta_adv_true

            -

            3 * theta_diff_true
        )

        for i in range(1,nx-1):

            for j in range(1,ny-1):

                center = idx(i,j)

                T_true_struct[
                    center,
                    center
                ] = theta_self_true

                # directional transport

                T_true_struct[
                    center,
                    idx(i-1,j)
                ] = theta_adv_true

                # diffusion

                T_true_struct[
                    center,
                    idx(i+1,j)
                ] = theta_diff_true

                T_true_struct[
                    center,
                    idx(i,j+1)
                ] = theta_diff_true

                T_true_struct[
                    center,
                    idx(i,j-1)
                ] = theta_diff_true

    

        u = np.zeros((nx,ny))

        u[20:25,10:15] = 10

        fields_over_time = []

        fields_over_time.append(
            u.copy()
        )

        nt = 100

        for t in range(nt):

            u_vec = u.reshape(-1)

            u_next = np.matmul(

                T_true_struct,

                u_vec
            )

            u = u_next.reshape(nx,ny)

            fields_over_time.append(
                u.copy()
            )

 

        U_t = []

        U_next = []

        for t in range(len(fields_over_time)-1):

            current_field = fields_over_time[t]

            next_field = fields_over_time[t+1]

            U_t.append(

                current_field.reshape(-1)
            )

            U_next.append(

                next_field.reshape(-1)
            )

        U_t = np.array(U_t)

        U_next = np.array(U_next)

 

        U_t_torch = torch.tensor(

            U_t,

            dtype=torch.float32
        ).to(device)

        U_next_torch = torch.tensor(

            U_next,

            dtype=torch.float32
        ).to(device)

   

        E_true_struct = []

        for t in range(len(fields_over_time)-1):

            exposure = np.matmul(

                T_true_struct,

                fields_over_time[t].reshape(-1)
            )

            E_true_struct.append(
                exposure
            )

        E_true_struct = np.array(
            E_true_struct
        )

    

        for sigma in sigma_grid:

            print()

            print("Sigma:", sigma)

 

            W_gaussian = np.exp(

                -(dist_matrix**2)/(2*sigma**2)
            )

            W_gaussian = (

                W_gaussian

                /

                W_gaussian.sum(
                    axis=1,
                    keepdims=True
                )
            )

    
            E_gaussian = []

            for t in range(len(fields_over_time)-1):

                exposure = np.matmul(

                    W_gaussian,

                    fields_over_time[t].reshape(-1)
                )

                E_gaussian.append(
                    exposure
                )

            E_gaussian = np.array(
                E_gaussian
            )

     

            gaussian_rmse = np.sqrt(

                np.mean(

                    (
                        E_true_struct

                        -

                        E_gaussian
                    )**2
                )
            )

            gaussian_corr = np.corrcoef(

                T_true_struct.flatten(),

                W_gaussian.flatten()
            )[0,1]

            gaussian_cond = cond(

                W_gaussian + 1e-6*np.eye(N)
            )
 

            singular_vals = np.linalg.svd(

                W_gaussian,

                compute_uv=False
            )

            effective_rank = np.sum(

                singular_vals > 1e-3
            )
 

            for n_monitors in monitor_grid:

                print(

                    "Monitors:",
                    n_monitors
                )

           

                np.random.seed(123)

                monitor_idx = np.random.choice(

                    N,

                    size=n_monitors,

                    replace=False
                )

           

                theta_self = nn.Parameter(
                    torch.tensor(0.7).to(device)
                )

                theta_adv = nn.Parameter(
                    torch.tensor(0.2).to(device)
                )

                theta_diff = nn.Parameter(
                    torch.tensor(0.05).to(device)
                )

                optimizer = optim.Adam(

                    [
                        theta_self,
                        theta_adv,
                        theta_diff
                    ],

                    lr=0.01
                )

     
                center_idx = []
                left_idx = []
                right_idx = []
                up_idx = []
                down_idx = []

                for i in range(1,nx-1):

                    for j in range(1,ny-1):

                        center = idx(i,j)

                        center_idx.append(center)

                        left_idx.append(idx(i-1,j))

                        right_idx.append(idx(i+1,j))

                        up_idx.append(idx(i,j+1))

                        down_idx.append(idx(i,j-1))

                center_idx = torch.tensor(center_idx).to(device)

                left_idx = torch.tensor(left_idx).to(device)

                right_idx = torch.tensor(right_idx).to(device)

                up_idx = torch.tensor(up_idx).to(device)

                down_idx = torch.tensor(down_idx).to(device)

       

                epochs = 300

                for epoch in range(epochs):

                    optimizer.zero_grad()

                    T_struct = torch.zeros(

                        (N,N),

                        device=device
                    )

                    T_struct[
                        center_idx,
                        center_idx
                    ] = torch.relu(theta_self)

                    T_struct[
                        center_idx,
                        left_idx
                    ] = torch.relu(theta_adv)

                    T_struct[
                        center_idx,
                        right_idx
                    ] = torch.relu(theta_diff)

                    T_struct[
                        center_idx,
                        up_idx
                    ] = torch.relu(theta_diff)

                    T_struct[
                        center_idx,
                        down_idx
                    ] = torch.relu(theta_diff)

                    

                    row_sums = (

                        T_struct.sum(
                            dim=1,
                            keepdim=True
                        )

                        +

                        1e-6
                    )

                    T_struct = (

                        T_struct

                        /

                        row_sums
                    )

               

                    U_pred = torch.matmul(

                        U_t_torch,

                        T_struct.T
                    )

                    U_pred_sparse = U_pred[:,monitor_idx]

                    U_next_sparse = U_next_torch[:,monitor_idx]

                   

                    loss = torch.mean(

                        (
                            U_pred_sparse

                            -

                            U_next_sparse
                        )**2
                    )

                    loss.backward()

                    optimizer.step()

          

                T_final = (

                    T_struct

                    .detach()

                    .cpu()

                    .numpy()
                )

           

                E_learned = []

                for t in range(len(fields_over_time)-1):

                    exposure = np.matmul(

                        T_final,

                        fields_over_time[t].reshape(-1)
                    )

                    E_learned.append(
                        exposure
                    )

                E_learned = np.array(
                    E_learned
                )

             

                learned_rmse = np.sqrt(

                    np.mean(

                        (
                            E_true_struct

                            -

                            E_learned
                        )**2
                    )
                )

                learned_corr = np.corrcoef(

                    T_true_struct.flatten(),

                    T_final.flatten()
                )[0,1]

                learned_cond = cond(

                    T_final + 1e-6*np.eye(N)
                )
 

                structured_results.append({

                    "theta_adv_true":
                        theta_adv_true,

                    "theta_diff_true":
                        theta_diff_true,

                    "sigma":
                        sigma,

                    "monitors":
                        n_monitors,

                    "gaussian_rmse":
                        gaussian_rmse,

                    "learned_rmse":
                        learned_rmse,

                    "gaussian_corr":
                        gaussian_corr,

                    "learned_corr":
                        learned_corr,

                    "gaussian_condition":
                        gaussian_cond,

                    "learned_condition":
                        learned_cond,

                    "effective_rank":
                        effective_rank,

                    "theta_self_est":
                        theta_self.item(),

                    "theta_adv_est":
                        theta_adv.item(),

                    "theta_diff_est":
                        theta_diff.item()
                })
 

structured_df = pd.DataFrame(
    structured_results
)

print()

print(structured_df.head())

 

structured_df.to_csv(

    "structured_transport_sensitivity.csv",

    index=False
)

print()

print("Saved:")
print("structured_transport_sensitivity.csv")


# In[7]:


import pandas as pd

structured_df = pd.DataFrame(
    structured_results
)

print(structured_df.head())

structured_df.to_csv(
    "structured_transport_sensitivity.csv",
    index=False
)

print("Saved!")


# In[8]:


structured_df.sort_values(
    "learned_corr",
    ascending=False
)


# In[9]:


structured_df[
    [
        "theta_adv_true",
        "theta_diff_true",
        "sigma",
        "monitors",
        "gaussian_corr",
        "learned_corr"
    ]
]


# In[ ]:


misspec_results = []

 

sigma_grid = [2,16]

monitor_grid = [100,10]

 

theta_adv_field = np.zeros((nx,ny))

theta_diff_field = np.zeros((nx,ny))

for i in range(nx):

    for j in range(ny):

    

        if i < nx//2:

            theta_adv_field[i,j] = 0.1

            theta_diff_field[i,j] = 0.01

  

        else:

            theta_adv_field[i,j] = 0.6

            theta_diff_field[i,j] = 0.10

 

T_true_hetero = np.zeros((N,N))

for i in range(1,nx-1):

    for j in range(1,ny-1):

        center = idx(i,j)

        theta_adv_true = theta_adv_field[i,j]

        theta_diff_true = theta_diff_field[i,j]

        theta_self_true = (

            1

            -

            theta_adv_true

            -

            3 * theta_diff_true
        )

        T_true_hetero[
            center,
            center
        ] = theta_self_true

        T_true_hetero[
            center,
            idx(i-1,j)
        ] = theta_adv_true

        T_true_hetero[
            center,
            idx(i+1,j)
        ] = theta_diff_true

        T_true_hetero[
            center,
            idx(i,j+1)
        ] = theta_diff_true

        T_true_hetero[
            center,
            idx(i,j-1)
        ] = theta_diff_true
 

u = np.zeros((nx,ny))

u[20:25,10:15] = 10

fields_over_time = []

fields_over_time.append(
    u.copy()
)

nt = 100

for t in range(nt):

    u_vec = u.reshape(-1)

    u_next = np.matmul(

        T_true_hetero,

        u_vec
    )

    u = u_next.reshape(nx,ny)

    fields_over_time.append(
        u.copy()
    )

 
U_t = []

U_next = []

for t in range(len(fields_over_time)-1):

    U_t.append(

        fields_over_time[t].reshape(-1)
    )

    U_next.append(

        fields_over_time[t+1].reshape(-1)
    )

U_t = np.array(U_t)

U_next = np.array(U_next)

U_t_torch = torch.tensor(

    U_t,

    dtype=torch.float32
).to(device)

U_next_torch = torch.tensor(

    U_next,

    dtype=torch.float32
).to(device)

 

E_true = []

for t in range(len(fields_over_time)-1):

    exposure = np.matmul(

        T_true_hetero,

        fields_over_time[t].reshape(-1)
    )

    E_true.append(
        exposure
    )

E_true = np.array(E_true)

 

for sigma in sigma_grid:

    print()

 

    print("SIGMA:", sigma)

 

 

    W_gaussian = np.exp(

        -(dist_matrix**2)/(2*sigma**2)
    )

    W_gaussian = (

        W_gaussian

        /

        W_gaussian.sum(
            axis=1,
            keepdims=True
        )
    )

    gaussian_corr = np.corrcoef(

        T_true_hetero.flatten(),

        W_gaussian.flatten()
    )[0,1]

 

    for n_monitors in monitor_grid:

        print()

        print("Monitors:", n_monitors)

        np.random.seed(123)

        monitor_idx = np.random.choice(

            N,

            size=n_monitors,

            replace=False
        )

 

        theta_self = nn.Parameter(
            torch.tensor(0.7).to(device)
        )

        theta_adv = nn.Parameter(
            torch.tensor(0.2).to(device)
        )

        theta_diff = nn.Parameter(
            torch.tensor(0.05).to(device)
        )

        optimizer = optim.Adam(

            [
                theta_self,
                theta_adv,
                theta_diff
            ],

            lr=0.01
        )

 

        center_idx = []
        left_idx = []
        right_idx = []
        up_idx = []
        down_idx = []

        for i in range(1,nx-1):

            for j in range(1,ny-1):

                center = idx(i,j)

                center_idx.append(center)

                left_idx.append(idx(i-1,j))

                right_idx.append(idx(i+1,j))

                up_idx.append(idx(i,j+1))

                down_idx.append(idx(i,j-1))

        center_idx = torch.tensor(center_idx).to(device)

        left_idx = torch.tensor(left_idx).to(device)

        right_idx = torch.tensor(right_idx).to(device)

        up_idx = torch.tensor(up_idx).to(device)

        down_idx = torch.tensor(down_idx).to(device)

   

        epochs = 300

        for epoch in range(epochs):

            optimizer.zero_grad()

            T_struct = torch.zeros(

                (N,N),

                device=device
            )

            T_struct[
                center_idx,
                center_idx
            ] = torch.relu(theta_self)

            T_struct[
                center_idx,
                left_idx
            ] = torch.relu(theta_adv)

            T_struct[
                center_idx,
                right_idx
            ] = torch.relu(theta_diff)

            T_struct[
                center_idx,
                up_idx
            ] = torch.relu(theta_diff)

            T_struct[
                center_idx,
                down_idx
            ] = torch.relu(theta_diff)

            row_sums = (

                T_struct.sum(
                    dim=1,
                    keepdim=True
                )

                +

                1e-6
            )

            T_struct = (

                T_struct

                /

                row_sums
            )

            U_pred = torch.matmul(

                U_t_torch,

                T_struct.T
            )

            U_pred_sparse = U_pred[:,monitor_idx]

            U_next_sparse = U_next_torch[:,monitor_idx]

            loss = torch.mean(

                (
                    U_pred_sparse

                    -

                    U_next_sparse
                )**2
            )

            loss.backward()

            optimizer.step()

        

        T_final = (

            T_struct

            .detach()

            .cpu()

            .numpy()
        )

        learned_corr = np.corrcoef(

            T_true_hetero.flatten(),

            T_final.flatten()
        )[0,1]

        learned_cond = cond(

            T_final + 1e-6*np.eye(N)
        )

   

        misspec_results.append({

            "sigma":
                sigma,

            "monitors":
                n_monitors,

            "gaussian_corr":
                gaussian_corr,

            "learned_corr":
                learned_corr,

            "learned_condition":
                learned_cond,

            "theta_self_est":
                theta_self.item(),

            "theta_adv_est":
                theta_adv.item(),

            "theta_diff_est":
                theta_diff.item()
        })

 

misspec_df = pd.DataFrame(
    misspec_results
)

print()

print(misspec_df)

misspec_df.to_csv(

    "stage3_misspecification.csv",

    index=False
)

print()

print("Saved:")
print("stage3_misspecification.csv")


# In[ ]:


import statsmodels.api as sm
np.random.seed(123)

 
A = np.random.binomial(

    1,

    0.5,

    N
)
 

E_true = np.matmul(

    T_true_hetero,

    A
)

 

beta_direct = 1.0

beta_spill = 2.0

Y = (

    beta_direct * A

    +

    beta_spill * E_true

    +

    np.random.normal(
        0,
        0.5,
        N
    )
)
 

sigma_grid = [2,4,8,16]

causal_results = []

for sigma in sigma_grid:

 

    W_gaussian = np.exp(

        -(dist_matrix**2)/(2*sigma**2)
    )

    W_gaussian = (

        W_gaussian

        /

        W_gaussian.sum(
            axis=1,
            keepdims=True
        )
    )

    E_gaussian = np.matmul(

        W_gaussian,

        A
    )

 

    X = np.column_stack([

        np.ones(N),

        A,

        E_gaussian
    ])

    model = sm.OLS(

        Y,

        X
    ).fit()

    beta_hat = model.params[2]

    causal_results.append({

        "sigma":
            sigma,

        "estimated_spillover":
            beta_hat,

        "bias":
            beta_hat - beta_spill
    })

 

causal_df = pd.DataFrame(
    causal_results
)

print()

print(causal_df)

causal_df.to_csv(

    "causal_mapping_bias.csv",

    index=False
)

print()

print("Saved:")
print("causal_mapping_bias.csv")


# In[ ]:




