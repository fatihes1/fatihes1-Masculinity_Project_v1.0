import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

survey = pd.read_csv("masculinity.csv")


#print(survey.columns)
print(len(survey))
#print(survey["q0007_0001"].value_counts())
#print(survey.head())

# Mapping Data
cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003",
    "q0007_0004","q0007_0005", "q0007_0006", "q0007_0007",
    "q0007_0008", "q0007_0009","q0007_0010", "q0007_0011"]

for col in cols_to_map:
    survey[col] = survey[col].map(
    {"Never, and not open to it": 0,
    "Never, but open to it": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4})

#print(survey['q0007_0001'].value_counts())

# Plotting Data
#plt.scatter(survey["q0007_0001"], survey["q0007_0002"], alpha = 0.1)
#plt.xlabel("Ask a friend for professional advice")
#plt.ylabel("Ask a friend for personal advice")
#plt.show()

#Build the KMeans Model


rows_to_cluster = survey.dropna(subset = [
    "q0007_0001",
    "q0007_0002",
    "q0007_0003",
    "q0007_0004",
    "q0007_0005",
    "q0007_0008",
    "q0007_0009"])

classifier = KMeans(n_clusters = 2)
classifier.fit(rows_to_cluster[[
    "q0007_0001", "q0007_0002", 
    "q0007_0003", "q0007_0004", 
    "q0007_0005", "q0007_0008", 
    "q0007_0009"]])
#print(classifier.cluster_centers_)
print(classifier.labels_)

cluster_zero_indices = []
cluster_one_indices = []
for i in range(len(classifier.labels_)):
    if classifier.labels_[i] == 0:
        cluster_zero_indices.append(i)
    elif classifier.labels_[i] == 1:
        cluster_one_indices.append(i)
        
#print(cluster_zero_indices)

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

#print(cluster_zero_df['educ4'].value_counts()/len(cluster_zero_df))
#print(cluster_one_df['educ4'].value_counts()/len(cluster_one_df))




