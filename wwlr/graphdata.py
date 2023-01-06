import networkx as nx
from typing import List
import os

import pickle

from . import retrieve_binary_filenames, \
    retrieve_graph_filenames


class Data:
    def __init__(self):
        self.graphs = None # list of Nx.digraph
        self.names = None # names of graphs
        self.labels = None # pandas dataframe with multiple columns

    def create_graph_generator(self, data_path, gexf):
        if gexf:
            filenames = retrieve_graph_filenames(data_path)
            self.graphs = (nx.read_gexf(f) for f in filenames)
        else:
            filenames = retrieve_binary_filenames(data_path)
            self.graphs = (pickle.load(open(f, 'rb')) for f in filenames)

        self.names = [os.path.splitext(os.path.basename(f))[0] for f in filenames]