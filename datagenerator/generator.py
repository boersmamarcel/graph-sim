import networkx as nx
import pickle
import pandas as pd
import os
import tqdm 
import numpy as np

class DataGenerator:

    def __init__(self, path, n):
        print("Data geneartor")

        self.n = n
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)
        

    def threeClasses(self, number_of_graphs):
        labels = []
        for i in tqdm.tqdm(range(number_of_graphs)):
            g1 = nx.complete_graph(self.n)
            # random values
            for g in g1.nodes():
                g1.nodes[g]['volume'] = np.random.uniform(0.1, 100)

            with open(os.path.join(self.path, str(i) + "_1.pickle"), 'wb') as f:
                pickle.dump(g1, f)
            labels.append(1)

            g2 = nx.barabasi_albert_graph(self.n, 5)
            for g in g2.nodes():
                g2.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            labels.append(2)
            with open(os.path.join(self.path, str(i) + "_2.pickle"), 'wb') as f2:
                pickle.dump(g2, f2)

            g3 = nx.erdos_renyi_graph(self.n, 0.2)
            labels.append(3)
            for g in g3.nodes():
                g3.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            with open(os.path.join(self.path, str(i) + "_3.pickle"), 'wb') as f3:
                pickle.dump(g3, f3)
                
        df = pd.DataFrame(labels) 
        df.to_csv(os.path.join(self.path, 'labels.tsv'), sep='\t', index=False)


    def smallClasses(self, number_of_graphs):
        labels = []
        for i in tqdm.tqdm(range(number_of_graphs)):
            g1 = nx.fast_gnp_random_graph(self.n, 0.3)
            # random values
            for g in g1.nodes():
                g1.nodes[g]['volume'] = np.random.uniform(0.1, 100)

            with open(os.path.join(self.path, str(i) + "_1.pickle"), 'wb') as f:
                pickle.dump(g1, f)
            labels.append(1)

            g2 = nx.barabasi_albert_graph(self.n, 5)
            for g in g2.nodes():
                g2.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            labels.append(2)
            with open(os.path.join(self.path, str(i) + "_2.pickle"), 'wb') as f2:
                pickle.dump(g2, f2)

            g3 = nx.watts_strogatz_graph(self.n, 3, 0.2)
            labels.append(3)
            for g in g3.nodes():
                g3.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            with open(os.path.join(self.path, str(i) + "_3.pickle"), 'wb') as f3:
                pickle.dump(g3, f3)
                
        df = pd.DataFrame(labels) 
        df.to_csv(os.path.join(self.path, 'labels.tsv'), sep='\t', index=False)

    def sameClassDifferentSettingsRandom(self, number_of_graphs):
        labels = []
        for i in tqdm.tqdm(range(number_of_graphs)):
            g1 = nx.fast_gnp_random_graph(self.n, 0.1)
            # random values
            for g in g1.nodes():
                g1.nodes[g]['volume'] = np.random.uniform(0.1, 100)

            with open(os.path.join(self.path, str(i) + "_1.pickle"), 'wb') as f:
                pickle.dump(g1, f)
            labels.append(1)

            g2 = nx.fast_gnp_random_graph(self.n, 0.4)
            for g in g2.nodes():
                g2.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            labels.append(2)
            with open(os.path.join(self.path, str(i) + "_2.pickle"), 'wb') as f2:
                pickle.dump(g2, f2)

            g3 = nx.fast_gnp_random_graph(self.n, 0.8)
            labels.append(3)
            for g in g3.nodes():
                g3.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            with open(os.path.join(self.path, str(i) + "_3.pickle"), 'wb') as f3:
                pickle.dump(g3, f3)
                
        df = pd.DataFrame(labels) 
        df.to_csv(os.path.join(self.path, 'labels.tsv'), sep='\t', index=False)

    def sameClassDifferentSettingsBarabasi(self, number_of_graphs):
        labels = []
        for i in tqdm.tqdm(range(number_of_graphs)):
            g1 = nx.barabasi_albert_graph(self.n, 5)
            # random values
            for g in g1.nodes():
                g1.nodes[g]['volume'] = np.random.uniform(0.1, 100)

            with open(os.path.join(self.path, str(i) + "_1.pickle"), 'wb') as f:
                pickle.dump(g1, f)
            labels.append(1)

            g2 = nx.barabasi_albert_graph(self.n, 10)
            for g in g2.nodes():
                g2.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            labels.append(2)
            with open(os.path.join(self.path, str(i) + "_2.pickle"), 'wb') as f2:
                pickle.dump(g2, f2)

            g3 = nx.barabasi_albert_graph(self.n, 15)
            labels.append(3)
            for g in g3.nodes():
                g3.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            with open(os.path.join(self.path, str(i) + "_3.pickle"), 'wb') as f3:
                pickle.dump(g3, f3)
                
        df = pd.DataFrame(labels) 
        df.to_csv(os.path.join(self.path, 'labels.tsv'), sep='\t', index=False)

    def sameClassDifferentSettingsWattsStrogatz(self, number_of_graphs):
        labels = []
        for i in tqdm.tqdm(range(number_of_graphs)):
            g1 = nx.watts_strogatz_graph(self.n, 5, 0.2)
            # random values
            for g in g1.nodes():
                g1.nodes[g]['volume'] = np.random.uniform(0.1, 100)

            with open(os.path.join(self.path, str(i) + "_1.pickle"), 'wb') as f:
                pickle.dump(g1, f)
            labels.append(1)

            g2 = nx.watts_strogatz_graph(self.n, 10, 0.4)
            for g in g2.nodes():
                g2.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            labels.append(2)
            with open(os.path.join(self.path, str(i) + "_2.pickle"), 'wb') as f2:
                pickle.dump(g2, f2)

            g3 = nx.watts_strogatz_graph(self.n, 15, 0.8)
            labels.append(3)
            for g in g3.nodes():
                g3.nodes[g]['volume'] = np.random.uniform(0.1, 100)
            with open(os.path.join(self.path, str(i) + "_3.pickle"), 'wb') as f3:
                pickle.dump(g3, f3)
                
        df = pd.DataFrame(labels) 
        df.to_csv(os.path.join(self.path, 'labels.tsv'), sep='\t', index=False)