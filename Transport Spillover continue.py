#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd

model_data = pd.read_csv(
    "transport_full_dataset.csv"
)

print(model_data.head())

print(model_data.columns)


# In[17]:


coords_df = pd.read_csv(
    "county_coordinates.csv"
)

model_data = model_data.merge(
    coords_df,
    on="county_name",
    how="left"
)

print(
    model_data[
        ["county_name","latitude","longitude"]
    ].head()
)


# In[18]:


print(model_data.shape)

print(
    model_data.isna().sum()
)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

 

model_data = pd.read_csv(
    "transport_full_dataset.csv"
)

coords_df = pd.read_csv(
    "county_coordinates.csv"
)

model_data = model_data.merge(
    coords_df,
    on="county_name",
    how="left"
)

 
 

county_coords = (
    model_data[
        ["county_name","latitude","longitude"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

coords = county_coords[
    ["latitude","longitude"]
].values

 

def build_gaussian_operator(
    coords,
    sigma
):

    D = cdist(coords, coords)

    W = np.exp(
        -(D**2)/(2*sigma**2)
    )

    np.fill_diagonal(W,0)

    row_sums = W.sum(axis=1)

    row_sums[row_sums==0] = 1

    W = W / row_sums[:,None]

    return W

 

sigmas = [1,3,5,10]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,10)
)

axes = axes.flatten()

for idx, sigma in enumerate(sigmas):

    W = build_gaussian_operator(
        coords,
        sigma
    )

    eigvals = np.linalg.eigvals(W)

    eigvals = np.sort(
        np.abs(eigvals)
    )[::-1]

    axes[idx].plot(
        eigvals,
        linewidth=2
    )

    axes[idx].set_title(
        f"$\\sigma={sigma}$"
    )

    axes[idx].set_xlabel(
        "Eigenvalue Index"
    )

    axes[idx].set_ylabel(
        "|Eigenvalue|"
    )

plt.tight_layout()

plt.savefig(
    "Figure_Eigenspectrum.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

 

sigma_grid = np.linspace(1,10,40)

condition_results = []

for sigma in sigma_grid:

    W = build_gaussian_operator(
        coords,
        sigma
    )

    cond_num = np.linalg.cond(W)

    condition_results.append(
        {
            "sigma": sigma,
            "condition_number": cond_num
        }
    )

condition_results = pd.DataFrame(
    condition_results
)

 

plt.figure(figsize=(7,5))

plt.plot(
    condition_results["sigma"],
    condition_results["condition_number"],
    linewidth=2
)

plt.xlabel("$\\sigma$")
plt.ylabel("Condition Number")

plt.tight_layout()

plt.savefig(
    "Figure_ConditionNumbers.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(condition_results.head())


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from sklearn.linear_model import LinearRegression

 

model_data = pd.read_csv(
    "transport_full_dataset.csv"
)

coords_df = pd.read_csv(
    "county_coordinates.csv"
)

model_data = model_data.merge(
    coords_df,
    on="county_name",
    how="left"
)

 

county_coords = (
    model_data[
        ["county_name","latitude","longitude"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

county_order = county_coords[
    "county_name"
].values

coords = county_coords[
    ["latitude","longitude"]
].values

 

def build_gaussian_operator(
    coords,
    sigma
):

    D = cdist(coords, coords)

    W = np.exp(
        -(D**2)/(2*sigma**2)
    )

    np.fill_diagonal(W,0)

    row_sums = W.sum(axis=1)

    row_sums[row_sums==0] = 1

    W = W / row_sums[:,None]

    return W

 

sigmas = np.linspace(1,10,50)

beta_results = []

dates = sorted(
    model_data["date_local"].unique()
)

for sigma in sigmas:

    W = build_gaussian_operator(
        coords,
        sigma
    )

    all_rows = []

    for d in dates:

        day_df = (
            model_data[
                model_data["date_local"] == d
            ]
            .copy()
        )

        day_df = (
            day_df
            .set_index("county_name")
            .reindex(county_order)
            .reset_index()
        )

        if day_df["pm25"].isna().any():
            continue

        y_today = day_df["pm25"].values

        spillover = W @ y_today

        day_df["spillover_temp"] = spillover

        all_rows.append(day_df)

    analysis_df = pd.concat(
        all_rows,
        axis=0
    )

    analysis_df = analysis_df.dropna()

    X = analysis_df[
        [
            "pm25_lag1",
            "spillover_temp"
        ]
    ].values

    y = analysis_df["pm25"].values

    model = LinearRegression()

    model.fit(X,y)

    beta_spill = model.coef_[1]

    beta_results.append(
        {
            "sigma": sigma,
            "beta_spillover": beta_spill
        }
    )

beta_results = pd.DataFrame(
    beta_results
)

 

plt.figure(figsize=(7,5))

plt.plot(
    beta_results["sigma"],
    beta_results["beta_spillover"],
    linewidth=2
)

plt.xlabel("$\\sigma$")
plt.ylabel("Estimated Spillover Coefficient")

plt.tight_layout()

plt.savefig(
    "Figure_PerturbationSensitivity.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(beta_results.head())


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

 
model_data = pd.read_csv(
    "transport_full_dataset.csv"
)

coords_df = pd.read_csv(
    "county_coordinates.csv"
)

model_data = model_data.merge(
    coords_df,
    on="county_name",
    how="left"
)

county_coords = (
    model_data[
        ["county_name","latitude","longitude"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

coords = county_coords[
    ["latitude","longitude"]
].values
 

def build_gaussian_operator(
    coords,
    sigma
):

    D = cdist(coords, coords)

    W = np.exp(
        -(D**2)/(2*sigma**2)
    )

    np.fill_diagonal(W,0)

    row_sums = W.sum(axis=1)

    row_sums[row_sums==0] = 1

    W = W / row_sums[:,None]

    return W

 

sigma_grid = np.linspace(1,10,40)

rank_results = []

for sigma in sigma_grid:

    W = build_gaussian_operator(
        coords,
        sigma
    )

    s = np.linalg.svd(
        W,
        compute_uv=False
    )

    effective_rank = np.sum(
        s > 0.01
    )

    condition_number = np.linalg.cond(W)

    rank_results.append(
        {
            "sigma": sigma,
            "effective_rank": effective_rank,
            "condition_number": condition_number
        }
    )

rank_results = pd.DataFrame(
    rank_results
)

 

plt.figure(figsize=(7,5))

plt.plot(
    rank_results["sigma"],
    rank_results["effective_rank"],
    linewidth=2
)

plt.xlabel("$\\sigma$")
plt.ylabel("Effective Rank")

plt.tight_layout()

plt.savefig(
    "Figure_EffectiveRank.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

 

plt.figure(figsize=(7,5))

plt.plot(
    rank_results["sigma"],
    rank_results["condition_number"],
    linewidth=2
)

plt.xlabel("$\\sigma$")
plt.ylabel("Condition Number")

plt.tight_layout()

plt.savefig(
    "Figure_RankCondition.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(rank_results.head())


# In[22]:


import pandas as pd

merged_data = pd.read_csv(
    "merged_data.csv"
)

print(merged_data.head())

print(merged_data.columns)


# In[23]:


merged_data["date_local"] = pd.to_datetime(

    merged_data["date_local"]
)


# In[24]:


print(merged_data.columns)


# In[ ]:


county_coords = pd.read_csv(
    "county_coordinates.csv"
)

print(county_coords.head())


# In[14]:


county_coords = pd.read_csv(
    "county_coordinates.csv"
)

print(county_coords.head())


# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from numpy.linalg import cond

import statsmodels.api as sm

 

county_coords = pd.read_csv(
    "county_coordinates.csv"
)

 

coords_rad = np.radians(

    county_coords[["latitude", "longitude"]]
)

distance_matrix = haversine_distances(
    coords_rad
)

# Earth radius in km
distance_matrix = distance_matrix * 6371

 

panel_data = merged_data.copy()

 

sigma_values = [25, 50, 75, 100, 150, 200, 300]

 

results_sigma = []

 

for s in sigma_values:
 

    gaussian_kernel = np.exp(

        -(distance_matrix ** 2) /

        (2 * s ** 2)
    )

    np.fill_diagonal(
        gaussian_kernel,
        0
    )

    

    row_sums = gaussian_kernel.sum(axis=1)

    gaussian_kernel = gaussian_kernel / row_sums[:, None]

 

    condition_number = cond(
        gaussian_kernel
    )

  

    spillover_all = []

    unique_dates = sorted(
        panel_data["date_local"].unique()
    )

    for d in unique_dates:

        day_data = panel_data[
            panel_data["date_local"] == d
        ].copy()

 

        day_data = county_coords.merge(

            day_data,

            on="county_name",

            how="left"
        )

      

        pm_vector = day_data["pm25"].fillna(0).values

    

        spillover = gaussian_kernel @ pm_vector

        day_data["gaussian_spillover"] = spillover

        spillover_all.append(
            day_data
        )

    spillover_data = pd.concat(
        spillover_all,
        axis=0
    )

   

    spillover_data = spillover_data.dropna(

        subset=[
            "pm25",
            "wind_speed",
            "temperature",
            "gaussian_spillover"
        ]
    )

    

    X = spillover_data[[
        "gaussian_spillover",
        "wind_speed",
        "temperature"
    ]]

    X = sm.add_constant(X)

    y = spillover_data["pm25"]

    model = sm.OLS(
        y,
        X
    ).fit()

    print(model.summary())

 

    pred = model.predict(X)

   

    rmse = np.sqrt(

        mean_squared_error(
            y,
            pred
        )
    )

 

    beta_transport = model.params[
        "gaussian_spillover"
    ]

 

    r_squared = model.rsquared
 

    results_sigma.append({

        "sigma": s,

        "beta_transport": beta_transport,

        "rmse": rmse,

        "condition_number": condition_number,

        "r_squared": r_squared
    })

 

results_sigma = pd.DataFrame(
    results_sigma
)

print("\nFINAL RESULTS\n")

print(results_sigma)

 

plt.figure(figsize=(7,5))

plt.plot(

    results_sigma["sigma"],

    results_sigma["beta_transport"],

    marker="o"
)

plt.xlabel("Sigma")

plt.ylabel("Transport Coefficient")

plt.title(
    "Sensitivity of Spillover Estimates to Gaussian Bandwidth"
)

plt.show()

 

plt.figure(figsize=(7,5))

plt.plot(

    results_sigma["sigma"],

    results_sigma["condition_number"],

    marker="o"
)

plt.xlabel("Sigma")

plt.ylabel("Condition Number")

plt.title(
    "Condition Number Across Gaussian Bandwidths"
)

plt.show()

 

plt.figure(figsize=(7,5))

plt.plot(

    results_sigma["sigma"],

    results_sigma["rmse"],

    marker="o"
)

plt.xlabel("Sigma")

plt.ylabel("RMSE")

plt.title(
    "RMSE Across Gaussian Bandwidths"
)

plt.show()

 

plt.figure(figsize=(7,5))

plt.plot(

    results_sigma["sigma"],

    results_sigma["r_squared"],

    marker="o"
)

plt.xlabel("Sigma")

plt.ylabel("R-squared")

plt.title(
    "R-squared Across Gaussian Bandwidths"
)

plt.show()

 

results_sigma.to_csv(

    "sigma_sensitivity_results.csv",

    index=False
)






# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import mean_squared_error

from numpy.linalg import cond

import statsmodels.api as sm


merged_data = pd.read_csv(
    "merged_data.csv"
)

county_coords = pd.read_csv(
    "county_coordinates.csv"
)

merged_data["date_local"] = pd.to_datetime(
    merged_data["date_local"]
)



county_coords = county_coords.sort_values(
    "county_name"
)

county_list = county_coords["county_name"].tolist()


coords_rad = np.radians(

    county_coords[
        ["latitude", "longitude"]
    ]
)

distance_matrix = haversine_distances(
    coords_rad
)


distance_matrix = distance_matrix * 6371


longitude_vector = county_coords[
    "longitude"
].values

direction_matrix = np.zeros_like(
    distance_matrix
)

for i in range(len(longitude_vector)):

    for j in range(len(longitude_vector)):


        if longitude_vector[j] < longitude_vector[i]:

            direction_matrix[i,j] = 1

        else:

            direction_matrix[i,j] = 0



sigma = 75

gaussian_kernel = np.exp(

    -(distance_matrix**2) /

    (2 * sigma**2)
)

np.fill_diagonal(
    gaussian_kernel,
    0
)

gaussian_kernel = (

    gaussian_kernel

    /

    gaussian_kernel.sum(axis=1)[:,None]
)



directional_kernel = (

    gaussian_kernel *

    direction_matrix
)

# avoid divide-by-zero

row_sums = directional_kernel.sum(axis=1)

row_sums[row_sums == 0] = 1

directional_kernel = (

    directional_kernel

    /

    row_sums[:,None]
)



cond_gaussian = cond(
    gaussian_kernel
)

cond_directional = cond(
    directional_kernel
)

print("\nCondition Numbers\n")

print("Gaussian:", cond_gaussian)

print("Directional:", cond_directional)



unique_dates = sorted(
    merged_data["date_local"].unique()
)

all_results = []

for d in unique_dates:

    day_data = merged_data[
        merged_data["date_local"] == d
    ].copy()



    day_data = county_coords.merge(

        day_data,

        on="county_name",

        how="left"
    )

    pm_vector = day_data[
        "pm25"
    ].fillna(0).values



    gaussian_spill = (

        gaussian_kernel @ pm_vector
    )



    directional_spill = (

        directional_kernel @ pm_vector
    )

    day_data["gaussian_spillover"] = (

        gaussian_spill
    )

    day_data["directional_spillover"] = (

        directional_spill
    )

    all_results.append(
        day_data
    )

spillover_data = pd.concat(
    all_results,
    axis=0
)



spillover_data = spillover_data.dropna(

    subset=[
        "pm25",
        "wind_speed",
        "temperature"
    ]
)



Xg = spillover_data[[
    "gaussian_spillover",
    "wind_speed",
    "temperature"
]]

Xg = sm.add_constant(Xg)

yg = spillover_data["pm25"]

model_gaussian = sm.OLS(
    yg,
    Xg
).fit()



print(model_gaussian.summary())



Xd = spillover_data[[
    "directional_spillover",
    "wind_speed",
    "temperature"
]]

Xd = sm.add_constant(Xd)

yd = spillover_data["pm25"]

model_directional = sm.OLS(
    yd,
    Xd
).fit()



print(model_directional.summary())



pred_g = model_gaussian.predict(Xg)

pred_d = model_directional.predict(Xd)

rmse_g = np.sqrt(

    mean_squared_error(
        yg,
        pred_g
    )
)

rmse_d = np.sqrt(

    mean_squared_error(
        yd,
        pred_d
    )
)



comparison = pd.DataFrame({

    "model": [

        "Gaussian",
        "Directional"
    ],

    "R_squared": [

        model_gaussian.rsquared,

        model_directional.rsquared
    ],

    "RMSE": [

        rmse_g,

        rmse_d
    ],

    "Condition_Number": [

        cond_gaussian,

        cond_directional
    ]
})



print(comparison)

plt.figure(figsize=(7,5))

plt.bar(

    comparison["model"],

    comparison["R_squared"]
)

plt.title(
    "R-squared Comparison"
)

plt.ylabel("R-squared")

plt.show()



spillover_data["resid_gaussian"] = (

    yg - pred_g
)

spillover_data["resid_directional"] = (

    yd - pred_d
)



plt.figure(figsize=(7,5))

plt.hist(

    spillover_data["resid_gaussian"],

    bins=30,

    alpha=0.7
)

plt.title(
    "Gaussian Residuals"
)

plt.show()

plt.figure(figsize=(7,5))

plt.hist(

    spillover_data["resid_directional"],

    bins=30,

    alpha=0.7
)

plt.title(
    "Directional Residuals"
)

plt.show()





comparison.to_csv(

    "directional_vs_gaussian.csv",

    index=False
)

spillover_data.to_csv(

    "spillover_results.csv",

    index=False
)


# In[ ]:


print(

    spillover_data[
        [
            "pm25",
            "wind_speed",
            "temperature",
            "gaussian_spillover",
            "directional_spillover"
        ]
    ].isna().sum()
)



# In[ ]:


alpha_values = [

    0,
    0.1,
    0.25,
    0.5,
    1
]

results_alpha = []

longitude_vector = county_coords[
    "longitude"
].values



for alpha in alpha_values:



    direction_matrix = np.zeros_like(
        distance_matrix
    )

    for i in range(len(longitude_vector)):

        for j in range(len(longitude_vector)):

            delta_lon = (

                longitude_vector[i]

                -

                longitude_vector[j]
            )



            direction_matrix[i,j] = np.exp(

                alpha *

                np.tanh(delta_lon)
            )


    directional_kernel = (

        gaussian_kernel

        *

        direction_matrix
    )



    row_sums = directional_kernel.sum(axis=1)

    row_sums[row_sums == 0] = 1

    directional_kernel = (

        directional_kernel

        /

        row_sums[:,None]
    )



    cond_directional = cond(
        directional_kernel
    )

    print("Condition Number:", cond_directional)



    unique_dates = sorted(
        merged_data["date_local"].unique()
    )

    all_results = []

    for d in unique_dates:

        day_data = merged_data[
            merged_data["date_local"] == d
        ].copy()

        day_data = county_coords.merge(

            day_data,

            on="county_name",

            how="left"
        )

        pm_vector = day_data[
            "pm25"
        ].fillna(0).values

        directional_spill = (

            directional_kernel @ pm_vector
        )

        day_data["directional_spillover"] = (

            directional_spill
        )

        all_results.append(
            day_data
        )

    spillover_data = pd.concat(
        all_results,
        axis=0
    )


    print(

        spillover_data[
            [
                "pm25",
                "wind_speed",
                "temperature",
                "directional_spillover"
            ]
        ].isna().sum()
    )



    Xd = spillover_data[[
        "directional_spillover",
        "wind_speed",
        "temperature"
    ]]

    Xd = sm.add_constant(Xd)

    yd = spillover_data["pm25"]

    model_directional = sm.OLS(
        yd,
        Xd,
        missing="drop"
    ).fit()

    pred_d = model_directional.predict(Xd)

    rmse_d = np.sqrt(

        mean_squared_error(
            yd,
            pred_d
        )
    )


    results_alpha.append({

        "alpha": alpha,

        "R_squared": model_directional.rsquared,

        "RMSE": rmse_d,

        "Condition_Number": cond_directional,

        "beta_transport":

            model_directional.params[
                "directional_spillover"
            ]
    })



results_alpha = pd.DataFrame(
    results_alpha
)


print(results_alpha)



plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["Condition_Number"],

    marker="o"
)

plt.xlabel("Alpha")

plt.ylabel("Condition Number")

plt.title(
    "Directional Strength vs Conditioning"
)

plt.show()



plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["R_squared"],

    marker="o"
)

plt.xlabel("Alpha")

plt.ylabel("R-squared")

plt.title(
    "Directional Strength vs Model Fit"
)

plt.show()


plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["beta_transport"],

    marker="o"
)

plt.xlabel("Alpha")

plt.ylabel("Transport Coefficient")

plt.title(
    "Directional Strength vs Spillover Estimate"
)

plt.show()


results_alpha.to_csv(

    "alpha_sensitivity_results.csv",

    index=False
)


# In[ ]:


model_data = spillover_data.dropna(

    subset=[

        "pm25",

        "wind_speed",

        "temperature",

        "directional_spillover"
    ]
)


print(model_data.shape)



Xd = model_data[[
    "directional_spillover",
    "wind_speed",
    "temperature"
]]

Xd = sm.add_constant(Xd)

yd = model_data["pm25"]

model_directional = sm.OLS(
    yd,
    Xd
).fit()

print(model_directional.summary())



pred_d = model_directional.predict(Xd)



rmse_d = np.sqrt(

    mean_squared_error(
        yd,
        pred_d
    )
)


# In[ ]:


alpha_values = [

    0,
    0.1,
    0.25,
    0.5,
    1
]

results_alpha = []

longitude_vector = county_coords[
    "longitude"
].values


for alpha in alpha_values:




    direction_matrix = np.zeros_like(
        distance_matrix
    )

    for i in range(len(longitude_vector)):

        for j in range(len(longitude_vector)):

            delta_lon = (

                longitude_vector[i]

                -

                longitude_vector[j]
            )

            direction_matrix[i,j] = np.exp(

                alpha *

                np.tanh(delta_lon)
            )


    directional_kernel = (

        gaussian_kernel

        *

        direction_matrix
    )


    row_sums = directional_kernel.sum(axis=1)

    row_sums[row_sums == 0] = 1

    directional_kernel = (

        directional_kernel

        /

        row_sums[:,None]
    )



    cond_directional = cond(
        directional_kernel
    )



    unique_dates = sorted(
        merged_data["date_local"].unique()
    )

    all_results = []

    for d in unique_dates:

        day_data = merged_data[
            merged_data["date_local"] == d
        ].copy()

        day_data = county_coords.merge(

            day_data,

            on="county_name",

            how="left"
        )

        pm_vector = day_data[
            "pm25"
        ].fillna(0).values

        directional_spill = (

            directional_kernel @ pm_vector
        )

        day_data["directional_spillover"] = (

            directional_spill
        )

        all_results.append(
            day_data
        )

    spillover_data = pd.concat(
        all_results,
        axis=0
    )



    model_data = spillover_data.dropna(

        subset=[

            "pm25",

            "wind_speed",

            "temperature",

            "directional_spillover"
        ]
    )



    Xd = model_data[[
        "directional_spillover",
        "wind_speed",
        "temperature"
    ]]

    Xd = sm.add_constant(Xd)

    yd = model_data["pm25"]

    model_directional = sm.OLS(
        yd,
        Xd
    ).fit()

    pred_d = model_directional.predict(Xd)

    rmse_d = np.sqrt(

        mean_squared_error(
            yd,
            pred_d
        )
    )



    results_alpha.append({

        "alpha": alpha,

        "R_squared": model_directional.rsquared,

        "RMSE": rmse_d,

        "Condition_Number": cond_directional,

        "beta_transport":

            model_directional.params[
                "directional_spillover"
            ]
    })



results_alpha = pd.DataFrame(
    results_alpha
)


plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["Condition_Number"],

    marker="o"
)

plt.xlabel("Alpha")

plt.ylabel("Condition Number")

plt.title(
    "Directional Strength vs Conditioning"
)

plt.show()



plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["R_squared"],

    marker="o"
)

plt.xlabel("Alpha")

plt.ylabel("R-squared")

plt.title(
    "Directional Strength vs Model Fit"
)

plt.show()


plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["beta_transport"],

    marker="o"
)

plt.xlabel("Alpha")

plt.ylabel("Transport Coefficient")

plt.title(
    "Directional Strength vs Spillover Estimate"
)

plt.show()


results_alpha.to_csv(

    "alpha_sensitivity_results.csv",

    index=False
)


# In[ ]:


import matplotlib.pyplot as plt



plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["Condition_Number"],

    marker="o",

    linewidth=2
)

plt.xlabel("Directional Strength (alpha)")

plt.ylabel("Condition Number")

plt.title(
    "Operator Conditioning Under Increasing Transport Asymmetry"
)

plt.yscale("log")

plt.grid(True)

plt.show()


plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["beta_transport"],

    marker="o",

    linewidth=2
)

plt.xlabel("Directional Strength (alpha)")

plt.ylabel("Estimated Spillover Coefficient")

plt.title(
    "Sensitivity of Spillover Estimates to Transport Directionality"
)

plt.grid(True)

plt.show()


plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["R_squared"],

    marker="o",

    linewidth=2
)

plt.xlabel("Directional Strength (alpha)")

plt.ylabel("R-squared")

plt.title(
    "Predictive Performance Under Directional Transport"
)

plt.grid(True)

plt.show()



plt.figure(figsize=(7,5))

plt.plot(

    results_alpha["alpha"],

    results_alpha["RMSE"],

    marker="o",

    linewidth=2
)

plt.xlabel("Directional Strength (alpha)")

plt.ylabel("RMSE")

plt.title(
    "Prediction Error Under Directional Transport"
)

plt.grid(True)

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt



nx = 50
ny = 50

dx = 1
dy = 1



nt = 150

dt = 0.05



# transport speed

v = 4.0

# diffusion

D = 0.03



u = np.zeros((nx, ny))

# pollution source

u[10:15, 10:15] = 100



history = []

history.append(u.copy())



for t in range(nt):

    u_new = u.copy()

    for i in range(1, nx-1):

        for j in range(1, ny-1):

          
            advect = (

                -v *

                (u[i,j] - u[i-1,j])

                / dx
            )

    

            diffuse = (

                D *

                (
                    u[i+1,j]
                    +
                    u[i-1,j]
                    +
                    u[i,j+1]
                    +
                    u[i,j-1]
                    -
                    4*u[i,j]
                )

                / dx**2
            )


            u_new[i,j] = (

                u[i,j]

                +

                dt *

                (
                    advect
                    +
                    diffuse
                )
            )

    u = u_new

    history.append(u.copy())

 

plt.figure(figsize=(7,6))

plt.imshow(
    u,
    origin="lower"
)

plt.colorbar()

plt.title(
    "Final Transport Field"
)

plt.show()


# In[ ]:


import pandas as pd

 

np.random.seed(123)

n_monitors = 100

monitor_x = np.random.randint(
    0,
    nx,
    n_monitors
)

monitor_y = np.random.randint(
    0,
    ny,
    n_monitors
)

 

observed_values = []

for k in range(n_monitors):

    x = monitor_x[k]

    y = monitor_y[k]

    value = u[x,y]

    # measurement noise

    noisy_value = (

        value

        +

        np.random.normal(
            0,
            2
        )
    )

    observed_values.append([

        x,
        y,
        noisy_value
    ])

observed_df = pd.DataFrame(

    observed_values,

    columns=[
        "x",
        "y",
        "observed_pm25"
    ]
)

print(observed_df.head())

 

plt.figure(figsize=(7,6))

plt.imshow(
    u,
    origin="lower"
)

plt.scatter(

    observed_df["y"],

    observed_df["x"],

    color="red",

    s=20
)

plt.title(
    "Sparse Monitor Locations"
)

plt.colorbar()

plt.show()


# In[ ]:


N = nx * ny

T_true = np.zeros((N, N))

 

def idx(i, j):

    return i * ny + j

 

for i in range(nx):

    for j in range(ny):

        center = idx(i,j)

        # self

        T_true[center, center] = 0.6

        # east transport

        if i+1 < nx:

            T_true[
                center,
                idx(i+1, j)
            ] = 0.25

        # diffusion neighbors

        if j+1 < ny:

            T_true[
                center,
                idx(i, j+1)
            ] = 0.075

        if j-1 >= 0:

            T_true[
                center,
                idx(i, j-1)
            ] = 0.075

 

from numpy.linalg import cond

print(

    "Condition Number:",

    cond(T_true)
)


# In[ ]:


from sklearn.metrics.pairwise import euclidean_distances

coords = []

for i in range(nx):

    for j in range(ny):

        coords.append([i,j])

coords = np.array(coords)

dist_matrix = euclidean_distances(
    coords
)

sigma = 2

T_gaussian = np.exp(

    -(dist_matrix**2)

    /

    (2*sigma**2)
)

# normalize

T_gaussian = (

    T_gaussian

    /

    T_gaussian.sum(axis=1)[:,None]
)

print(

    "Gaussian Condition Number:",

    cond(T_gaussian)
)


# In[ ]:


fro_error = (

    np.linalg.norm(
        T_true - T_gaussian,
        ord="fro"
    )

    /

    np.linalg.norm(
        T_true,
        ord="fro"
    )
)

print(

    "Relative Operator Error:",

    fro_error
)

 


# In[ ]:


from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import (

    RBF,
    WhiteKernel
)

 

X_obs = observed_df[[
    "x",
    "y"
]].values

y_obs = observed_df[
    "observed_pm25"
].values
 

kernel = (

    1.0 * RBF(length_scale=5.0)

    +

    WhiteKernel(noise_level=1)
)

 

gp = GaussianProcessRegressor(

    kernel=kernel,

    alpha=0.0
)

gp.fit(
    X_obs,
    y_obs
)

 

grid_points = []

for i in range(nx):

    for j in range(ny):

        grid_points.append([i,j])

grid_points = np.array(
    grid_points
)
 

pred_mean, pred_sd = gp.predict(

    grid_points,

    return_std=True
)

 

u_hat = pred_mean.reshape(nx, ny)

sd_hat = pred_sd.reshape(nx, ny)

 

plt.figure(figsize=(7,6))

plt.imshow(
    u,
    origin="lower"
)

plt.title(
    "True Transport Field"
)

plt.colorbar()

plt.show()

 

plt.figure(figsize=(7,6))

plt.imshow(
    u_hat,
    origin="lower"
)

plt.title(
    "GP-Reconstructed Field"
)

plt.colorbar()

plt.show()

 

plt.figure(figsize=(7,6))

plt.imshow(
    sd_hat,
    origin="lower"
)

plt.title(
    "Reconstruction Uncertainty"
)

plt.colorbar()

plt.show()

 

reconstruction_error = np.sqrt(

    np.mean(
        (u - u_hat)**2
    )
)

print(

    "Reconstruction RMSE:",

    reconstruction_error
)


# In[ ]:


boundary_points = []

# top + bottom boundaries

for i in range(nx):

    boundary_points.append([

        i / nx,
        0
    ])

    boundary_points.append([

        i / nx,
        1
    ])

# left + right boundaries

for j in range(ny):

    boundary_points.append([

        0,
        j / ny
    ])

    boundary_points.append([

        1,
        j / ny
    ])

boundary_points = np.array(
    boundary_points
)

 

boundary_torch = torch.tensor(

    boundary_points,

    dtype=torch.float32
).to(device)

print(

    "Boundary Shape:",

    boundary_torch.shape
)


# In[ ]:


epochs = 5000

loss_history = []

for epoch in range(epochs):

    optimizer.zero_grad()

 

    pred_obs = model(
        X_obs_torch
    )

    loss_data = torch.mean(

        (
            pred_obs

            -

            y_obs_torch
        )**2
    )

 

    u_pred = model(
        X_f_torch
    )

    grads = torch.autograd.grad(

        u_pred,

        X_f_torch,

        grad_outputs=torch.ones_like(u_pred),

        create_graph=True
    )[0]

    u_x = grads[:,0:1]

    u_y = grads[:,1:2]

 

    u_xx = torch.autograd.grad(

        u_x,

        X_f_torch,

        grad_outputs=torch.ones_like(u_x),

        create_graph=True
    )[0][:,0:1]

    u_yy = torch.autograd.grad(

        u_y,

        X_f_torch,

        grad_outputs=torch.ones_like(u_y),

        create_graph=True
    )[0][:,1:2]

 

    pde_residual = (

        v * u_x

        -

        D * (u_xx + u_yy)
    )

    loss_pde = torch.mean(

        pde_residual**2
    )

 

    boundary_pred = model(
        boundary_torch
    )

    loss_boundary = torch.mean(

        boundary_pred**2
    )

 

    loss = (

        loss_data

        +

        0.01 * loss_pde

        +

        0.1 * loss_boundary
    )

 

    loss.backward()

    optimizer.step()

 

    loss_history.append(
        loss.item()
    )

 

    if epoch % 500 == 0:

        print(

            "Epoch:", epoch,

            "Total:", round(loss.item(),4),

            "Data:", round(loss_data.item(),4),

            "PDE:", round(loss_pde.item(),4),

            "Boundary:", round(loss_boundary.item(),4)
        )


# In[ ]:


grid_points = []

for i in range(nx):

    for j in range(ny):

        grid_points.append([

            i / nx,

            j / ny
        ])

grid_points = np.array(
    grid_points
)

grid_torch = torch.tensor(

    grid_points,

    dtype=torch.float32
).to(device)

 

with torch.no_grad():

    pred_field = model(
        grid_torch
    ).cpu().numpy()

u_pinn = pred_field.reshape(
    nx,
    ny
)

 

plt.figure(figsize=(7,6))

plt.imshow(
    u,
    origin="lower"
)

plt.title(
    "True Transport Field"
)

plt.colorbar()

plt.show()

 

plt.figure(figsize=(7,6))

plt.imshow(
    u_pinn,
    origin="lower"
)

plt.title(
    "PINN-Reconstructed Field"
)

plt.colorbar()

plt.show()

 

error_field = np.abs(
    u - u_pinn
)

plt.figure(figsize=(7,6))

plt.imshow(
    error_field,
    origin="lower"
)

plt.title(
    "Absolute Reconstruction Error"
)

plt.colorbar()

plt.show()

 

rmse_pinn = np.sqrt(

    np.mean(
        (u - u_pinn)**2
    )
)

print(

    "PINN Reconstruction RMSE:",

    round(rmse_pinn,4)
)


# In[ ]:


u_x_grad = np.zeros_like(u_pinn)

u_y_grad = np.zeros_like(u_pinn)

 

for i in range(1, nx-1):

    for j in range(1, ny-1):

        u_x_grad[i,j] = (

            u_pinn[i+1,j]

            -

            u_pinn[i-1,j]

        ) / 2

        u_y_grad[i,j] = (

            u_pinn[i,j+1]

            -

            u_pinn[i,j-1]

        ) / 2

 

grad_mag = np.sqrt(

    u_x_grad**2

    +

    u_y_grad**2
)

 

plt.figure(figsize=(7,6))

plt.imshow(
    grad_mag,
    origin="lower"
)

plt.title(
    "PINN Gradient Magnitude"
)

plt.colorbar()

plt.show()


# In[ ]:


N = nx * ny

T_pinn = np.zeros((N,N))

 

def idx(i,j):

    return i * ny + j

 

for i in range(1, nx-1):

    for j in range(1, ny-1):

        center = idx(i,j)

 

        gx = u_x_grad[i,j]

        gy = u_y_grad[i,j]



        weights = []

        neighbors = [

            (i+1,j),

            (i-1,j),

            (i,j+1),

            (i,j-1)
        ]

        for ni,nj in neighbors:

            direction = np.array([

                ni - i,

                nj - j
            ])

            grad_vec = np.array([

                gx,

                gy
            ])

            transport_strength = np.abs(

                np.dot(
                    direction,
                    grad_vec
                )
            )

            weights.append(
                transport_strength
            )

        weights = np.array(weights)



        if weights.sum() > 0:

            weights = (

                weights

                /

                weights.sum()
            )

 

        for k,(ni,nj) in enumerate(neighbors):

            T_pinn[

                center,

                idx(ni,nj)

            ] = weights[k]
 

from numpy.linalg import cond

cond_pinn = cond(

    T_pinn + 1e-6*np.eye(N)
)

print(

    "PINN Operator Condition Number:",

    cond_pinn
)


# In[ ]:


fro_error_pinn = (

    np.linalg.norm(

        T_true

        -

        T_pinn,

        ord="fro"
    )

    /

    np.linalg.norm(
        T_true,
        ord="fro"
    )
)

print(

    "PINN Relative Operator Error:",

    round(fro_error_pinn,4)
)

 

corr_operator = np.corrcoef(

    T_true.flatten(),

    T_pinn.flatten()
)[0,1]

print(

    "Operator Correlation:",

    round(corr_operator,4)
)


# In[ ]:


fro_error_gaussian = (

    np.linalg.norm(

        T_true

        -

        T_gaussian,

        ord="fro"
    )

    /

    np.linalg.norm(
        T_true,
        ord="fro"
    )
)

corr_gaussian = np.corrcoef(

    T_true.flatten(),

    T_gaussian.flatten()
)[0,1]

 

comparison_results = pd.DataFrame({

    "Method":[

        "Gaussian",

        "PINN"
    ],

    "Relative_Operator_Error":[

        fro_error_gaussian,

        fro_error_pinn
    ],

    "Operator_Correlation":[

        corr_gaussian,

        corr_operator
    ]
})

print(comparison_results)


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
 
 

u_true = torch.tensor(

    u.reshape(-1,1),

    dtype=torch.float32
).to(device)

 

N = nx * ny

# learnable operator

T_param = nn.Parameter(

    0.01 * torch.randn(N,N)
)

optimizer_T = optim.Adam(

    [T_param],

    lr=0.001
)

 

local_mask = np.zeros((N,N))

def idx(i,j):

    return i * ny + j

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

 

epochs_operator = 3000

loss_history_operator = []

for epoch in range(epochs_operator):

    optimizer_T.zero_grad()

    
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

 

    u_pred = torch.matmul(

        T_norm,

        u_true
    )
 

    loss_reconstruct = torch.mean(

        (
            u_pred

            -

            u_true
        )**2
    )

 

    loss_sparse = torch.mean(
        torch.abs(T_norm)
    )

 

    loss = (

        loss_reconstruct

        +

        0.001 * loss_sparse
    )

    loss.backward()

    optimizer_T.step()

    loss_history_operator.append(
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

 

fro_error_direct = (

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

print(

    "Direct Operator Recovery Error:",

    round(fro_error_direct,4)
)

 
corr_direct = np.corrcoef(

    T_true.flatten(),

    T_learned.flatten()
)[0,1]

print(

    "Direct Operator Correlation:",

    round(corr_direct,4)
)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

 

nx = 50
ny = 50

dx = 1
dy = 1

 

D = 0.03

v_x = 4.0

dt = 0.01

nt = 100
 
 

u = np.zeros((nx,ny))

# localized source

u[20:25,10:15] = 10

 

fields_over_time = []

fields_over_time.append(
    u.copy()
)

 

for t in range(nt):

    u_new = u.copy()

    for i in range(1,nx-1):

        for j in range(1,ny-1):

 

            adv_x = (

                u[i,j]

                -

                u[i-1,j]

            ) / dx

 

            laplacian = (

                u[i+1,j]

                +

                u[i-1,j]

                +

                u[i,j+1]

                +

                u[i,j-1]

                -

                4*u[i,j]

            )

 

            u_new[i,j] = (

                u[i,j]

                -

                v_x * dt * adv_x

                +

                D * dt * laplacian
            )

    u = u_new.copy()

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


# In[ ]:


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

print(U_t.shape)

print(U_next.shape)


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

 

U_t_torch = torch.tensor(

    U_t,

    dtype=torch.float32
).to(device)

U_next_torch = torch.tensor(

    U_next,

    dtype=torch.float32
).to(device)

 

N = nx * ny

T_param = nn.Parameter(

    0.001 * torch.randn(N,N)
)

optimizer = optim.Adam(

    [T_param],

    lr=0.001
)

 

local_mask = np.zeros((N,N))

def idx(i,j):

    return i * ny + j

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
 

    loss_temporal = torch.mean(

        (
            U_pred

            -

            U_next_torch
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


# In[ ]:


T_true = np.zeros((N,N))

for i in range(1,nx-1):

    for j in range(1,ny-1):

        center = idx(i,j)

        # self

        T_true[center,center] = 0.7

        # directional transport

        T_true[center,idx(i+1,j)] = 0.25

        # diffusion

        T_true[center,idx(i,j+1)] = 0.025

        T_true[center,idx(i,j-1)] = 0.025


# In[ ]:


fro_error = (

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

print(

    "Relative Operator Error:",

    round(fro_error,4)
)

 

corr = np.corrcoef(

    T_true.flatten(),

    T_learned.flatten()
)[0,1]

print(

    "Operator Correlation:",

    round(corr,4)
)

 

from numpy.linalg import cond

condition_number = cond(

    T_learned + 1e-6*np.eye(N)
)

print(

    "Condition Number:",

    round(condition_number,4)
)


# In[ ]:





# In[ ]:


E_true = []

for t in range(len(fields_over_time)-1):

    current_field = fields_over_time[t]

    exposure_field = current_field.copy()

    E_true.append(

        exposure_field.reshape(-1)
    )

E_true = np.array(E_true)

print(E_true.shape)


# In[ ]:


from sklearn.metrics.pairwise import euclidean_distances

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

        E_true[t]
    )

    E_gaussian.append(
        exposure
    )

E_gaussian = np.array(
    E_gaussian
)


# In[ ]:


from sklearn.metrics.pairwise import euclidean_distances

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

        E_true[t]
    )

    E_gaussian.append(
        exposure
    )

E_gaussian = np.array(
    E_gaussian
)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

nx = 50
ny = 50

dx = 1
dy = 1

D = 0.03

v_x = 4.0

dt = 0.01

nt = 100

 

u = np.zeros((nx,ny))

u[20:25,10:15] = 10

 

fields_over_time = []

fields_over_time.append(
    u.copy()
)

 

for t in range(nt):

    u_new = u.copy()

    for i in range(1,nx-1):

        for j in range(1,ny-1):
 

            adv_x = (

                u[i,j]

                -

                u[i-1,j]

            ) / dx

          

            laplacian = (

                u[i+1,j]

                +

                u[i-1,j]

                +

                u[i,j+1]

                +

                u[i,j-1]

                -

                4*u[i,j]
            )

       

            u_new[i,j] = (

                u[i,j]

                -

                v_x * dt * adv_x

                +

                D * dt * laplacian
            )

    u = u_new.copy()

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


# In[ ]:


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

print(U_t.shape)

print(U_next.shape)


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)
 

U_t_torch = torch.tensor(

    U_t,

    dtype=torch.float32
).to(device)

U_next_torch = torch.tensor(

    U_next,

    dtype=torch.float32
).to(device)

 

N = nx * ny

 

T_param = nn.Parameter(

    0.001 * torch.randn(N,N)
)

optimizer = optim.Adam(

    [T_param],

    lr=0.001
)

 

local_mask = np.zeros((N,N))

def idx(i,j):

    return i * ny + j

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

 
    loss_temporal = torch.mean(

        (
            U_pred

            -

            U_next_torch
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


# In[ ]:


T_true = np.zeros((N,N))

for i in range(1,nx-1):

    for j in range(1,ny-1):

        center = idx(i,j)

        # self

        T_true[center,center] = 0.7

        # directional transport

        T_true[center,idx(i+1,j)] = 0.25

        # diffusion

        T_true[center,idx(i,j+1)] = 0.025

        T_true[center,idx(i,j-1)] = 0.025


# In[ ]:


fro_error = (

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

print(

    "Relative Operator Error:",

    round(fro_error,4)
)

 

corr = np.corrcoef(

    T_true.flatten(),

    T_learned.flatten()
)[0,1]

print(

    "Operator Correlation:",

    round(corr,4)
)

 
from numpy.linalg import cond

condition_number = cond(

    T_learned + 1e-6*np.eye(N)
)

print(

    "Condition Number:",

    round(condition_number,4)
)


# In[ ]:


E_learned = []

for t in range(len(fields_over_time)-1):

    exposure = np.matmul(

        T_learned,

        E_true[t]
    )

    E_learned.append(
        exposure
    )

E_learned = np.array(
    E_learned
)


# In[ ]:


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

print(

    "Gaussian Mapping RMSE:",

    round(rmse_gaussian,4)
)

print(

    "Learned Mapping RMSE:",

    round(rmse_learned,4)
)


# In[ ]:


from numpy.linalg import cond

cond_gaussian = cond(

    W_gaussian + 1e-6*np.eye(N)
)

cond_learned = cond(

    T_learned + 1e-6*np.eye(N)
)

print(

    "Gaussian Condition:",

    round(cond_gaussian,4)
)

print(

    "Learned Operator Condition:",

    round(cond_learned,4)
)


# In[ ]:


corr_gaussian = np.corrcoef(

    T_true.flatten(),

    W_gaussian.flatten()
)[0,1]

corr_learned = np.corrcoef(

    T_true.flatten(),

    T_learned.flatten()
)[0,1]

print(

    "Gaussian Correlation:",

    round(corr_gaussian,4)
)

print(

    "Learned Correlation:",

    round(corr_learned,4)
)


# In[ ]:





# In[ ]:




