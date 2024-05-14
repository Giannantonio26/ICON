import pickle
import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.models import BayesianNetwork

# Funzione che visualizza il grafo del Bayesian Network
def visualizeBayesianNetwork(bayesianNetwork: BayesianNetwork):
    G = nx.MultiDiGraph(bayesianNetwork.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=7,
        arrowstyle="->",
        edge_color="purple",
        connectionstyle="angle3,angleA=90,angleB=0",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()
    plt.clf()


# funzione per creare rete bayesiana
def bNetCreation(dataSet):
    #Ricerca della struttura ottimale
    hc_k2=HillClimbSearch(dataSet)
    k2_model=hc_k2.estimate(scoring_method='k2score')
    #Creazione della rete bayesiana
    model = BayesianNetwork(k2_model.edges())
    model.fit(dataSet,estimator=MaximumLikelihoodEstimator,n_jobs=-1)
    #Salvo la rete bayesiana su file
    with open('data/modello.pkl', 'wb') as output:
        pickle.dump(model, output)
    visualizeBayesianNetwork(model)
    return model