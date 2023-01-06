import urllib
import zipfile
import networkx as nx
import io
import collections
import numpy as np


def download_dataset(dataset_url, categorical=False, subset=None):
    resp = urllib.request.urlopen(dataset_url)
    unzipped = zipfile.ZipFile(io.BytesIO(resp.read()))

    g = nx.Graph()

    def path(suffix):
        paths = [n for n in unzipped.namelist() if n.endswith(suffix)]
        assert len(paths) <= 1
        return paths[0] if paths else None

    def open_file(suffix):
        return unzipped.open(path(suffix), 'r')

    try:
        if categorical:
            for i, line in enumerate(open_file('_node_labels.txt'), 1):
                g.add_node(i, label=int(line))
        else:
            for i, line in enumerate(open_file('_node_attributes.txt'), 1):
                line = line.decode('utf-8').replace(' ', '').split(',')
                attrs = {f'attr_{k}': float(v) for k, v in enumerate(line)}
                g.add_node(i, **attrs)
    except KeyError:
        pass

    edges = ((int(n) for n in line.split(b',')) for line in open_file('_A.txt'))

    try:
        labels = (int(line) for line in open_file('_edge_labels.txt'))
    except KeyError:
        labels = None

    if labels:
        for edge, label in zip(edges, labels):
            g.add_edge(*edge, label=label)
    else:
        for edge in edges:
            g.add_edge(*edge)

    gs = collections.defaultdict(list)

    for i, id in enumerate(open_file('_graph_indicator.txt'), 1):
        gs[int(id)].append(i)

    gs = {k: nx.Graph(g.subgraph(v)) for k, v in gs.items()}

    for i, label in enumerate(open_file('_graph_labels.txt'), 1):
        gs[i].graph['label'] = int(label)

    print('Loaded {} with {} graphs.'.format(dataset_url, len(gs)))

    def convert(g):
        return nx.convert_node_labels_to_integers(g, first_label=0)

    if subset:
        keys = np.random.choice(list(gs.keys()), np.int(np.floor(subset * len(list(gs.keys())))))
        return {k: convert(gs[k]) for k in keys}
    else:
        return {k: convert(v) for k, v in gs.items()}
