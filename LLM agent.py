#!/usr/bin/env python
# coding: utf-8

# pip install pandas numpy matplotlib seaborn scikit-learn openai ydata-profiling networkx

# pip install --ignore-installed --no-deps matplotlib==3.10.9

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from ydata_profiling import ProfileReport

from openai import OpenAI

import networkx as nx

 

client = OpenAI(
    api_key="YOUR_OPENAI_API_KEY"
)

 

df = pd.read_csv("combined.csv")

 

df = df[df["visit"] == 4].copy()

print(df.shape)

 

remove_keywords = [
    "cvl",
    "vload2",
    "esthivdate",
    "lg10cvl",
    "cvlfrhaart"
]

pattern = "|".join(remove_keywords)

keep_cols = ~df.columns.str.contains(
    pattern,
    case=False,
    regex=True
)

df = df.loc[:, keep_cols]

print(df.shape)

 

numeric_df = df.select_dtypes(include=np.number)

print(numeric_df.shape)
 

corr_matrix = numeric_df.corr().abs()

upper = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

remove_cols = [
    column
    for column in upper.columns
    if any(upper[column] > 0.95)
]

numeric_df = numeric_df.drop(columns=remove_cols)

print("Removed correlated variables:")
print(len(remove_cols))

print(numeric_df.shape)

 

numeric_df = numeric_df.dropna()

 

profile = ProfileReport(
    numeric_df,
    title="Biomedical Exploratory Analysis",
    explorative=True
)

profile.to_file("Biomedical_Exploratory_Report.html")

 

outcomes = [
    "aar",
    "eeaa",
    "peaa",
    "geaa",
    "dnamtladjage"
]

 

environmental_vars = [
    "Precipitation",
    "Temperature",
    "SPWPM2.5",
    "SBMPM2.5",
    "CO",
    "COAQI",
    "NO2",
    "NO2AQI",
    "Ozone",
    "OzoneAQI",
    "SO2",
    "SO2AQI"
]

environmental_vars = [
    x for x in environmental_vars
    if x in numeric_df.columns
]

 

immune_vars = [
    "lncd3",
    "lncd4",
    "lncd8",
    "lnnaivecd4",
    "lnnaivecd8",
    "lnsencd4",
    "lnsencd8",
    "lnactcd4",
    "lnactcd8"
]

immune_vars = [
    x for x in immune_vars
    if x in numeric_df.columns
]

 

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    numeric_df
)

 

pca = PCA(n_components=2)

pca_data = pca.fit_transform(
    scaled_data
)

 

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("PCA of Biomedical Data")

plt.tight_layout()

plt.savefig(
    "PCA_Biomedical.png",
    dpi=300
)

plt.show()

 

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(
    scaled_data
)

numeric_df["cluster"] = clusters

 

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=clusters
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("KMeans Clusters")

plt.tight_layout()

plt.savefig(
    "KMeans_Clusters.png",
    dpi=300
)

plt.show()

 

for outcome in outcomes:

    if outcome not in numeric_df.columns:
        continue

 

    y = numeric_df[outcome]

    X = numeric_df.drop(columns=outcomes + ["cluster"])

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        f"RF_{outcome}_importance.csv",
        index=False
    )

    top20 = importance_df.head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        top20["Feature"][::-1],
        top20["Importance"][::-1]
    )

    plt.title(
        f"Top Features for {outcome.upper()}"
    )

    plt.tight_layout()

    plt.savefig(
        f"RF_{outcome}_Top20.png",
        dpi=300
    )

    plt.show()

 

corr = numeric_df.corr()

threshold = 0.5

G = nx.Graph()

for i in corr.columns:

    for j in corr.columns:

        if i != j:

            value = corr.loc[i, j]

            if abs(value) > threshold:

                G.add_edge(
                    i,
                    j,
                    weight=value
                )

plt.figure(figsize=(14, 12))

pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=200,
    font_size=6
)

plt.title("Correlation Network")

plt.savefig(
    "Correlation_Network.png",
    dpi=300
)

plt.show()

 

summary_text = corr[
    outcomes
].sort_values(
    by=outcomes[0],
    ascending=False
).head(30).to_string()

prompt = f"""
You are a biomedical discovery agent.

Dataset context:
- HIV aging cohort
- Environmental exposures
- Immune markers
- Epigenetic aging outcomes

Generate:
1. Biological hypotheses
2. Candidate causal structures
3. Potential mediators
4. Potential subgroup effects
5. Environmental exposure patterns
6. Potential immune-environment interactions

Correlation summary:

{summary_text}

Focus on:
- inflammation
- immune senescence
- pollution exposure
- accelerated epigenetic aging

Do NOT claim causality.
Generate exploratory scientific hypotheses only.
"""

response = client.chat.completions.create(

    model="gpt-5.5",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],

    temperature=0.7
)

hypotheses = response.choices[0].message.content

print(hypotheses)

 

with open(
    "LLM_Hypotheses.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(hypotheses)

 
 
 

response_dag = client.chat.completions.create(

    model="gpt-5.5",

    messages=[
        {
            "role": "user",
            "content": prompt_dag
        }
    ],

    temperature=0.5
)

dag_text = response_dag.choices[0].message.content

print(dag_text)

 

with open(
    "LLM_DAG_Suggestions.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(dag_text)

 
 
 

