from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Funzione che calcola il numero di cluster ottimale per il dataset mediante il metodo del gomito
def regolaGomito(numerical_features_scaled):
    intertia = []
    #fisso un range di k da 1 a 10
    maxK=10
    for i in range(1, maxK):
        #eseguo il kmeans per ogni k, con 5 inizializzazioni diverse e con inizializzazione random. Prendo la migliore
        kmeans = KMeans(n_clusters=i,n_init=5,init='random')
        kmeans.fit(numerical_features_scaled)
        intertia.append(kmeans.inertia_)
    #mediante la libreria kneed trovo il k ottimale
    kl = KneeLocator(range(1, maxK), intertia, curve="convex", direction="decreasing")
    # Visualizza il grafico con la nota per il miglior k
    plt.plot(range(1, maxK), intertia, 'bx-')
    plt.scatter(kl.elbow, intertia[kl.elbow - 1], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Metodo del gomito per trovare il k ottimale')
    plt.legend()
    plt.show()
    return kl.elbow


#Funzione che esegue il kmeans sul dataset e restituisce le etichette e i centroidi
def calcolaCluster(dataSet):
    numerical_features = dataSet.select_dtypes(include=[np.number])
    #Standardizzazione delle feature numeriche
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    k=regolaGomito(numerical_features_scaled)
    km = KMeans(n_clusters=k,n_init=10,init='random')
    km = km.fit(numerical_features_scaled)
    etichette = km.labels_
    centroidi = km.cluster_centers_
    return etichette, centroidi


#Funzione che visualizza il grafico a torta per la distribuzione dei valori di differentialColumn
def getRatioChart(dataSet, differentialColumn, title):    
    counts = dataSet[differentialColumn].value_counts()
    # Etichette e colori per il grafico
    labels = counts.index.tolist()
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold', 'mediumorchid', 'lightsteelblue', 'lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue','lightgreen','lightcoral','lightpink','lightgrey','lightcyan','lightyellow','lightseagreen','lightsalmon','lightblue']
    #lunga lista di colori per evitare ripetizioni in caso di molti valori unici
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.legend(labels, loc='lower left', fontsize='small')
    plt.title(title)
    plt.show()