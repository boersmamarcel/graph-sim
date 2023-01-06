from .dissimilarity_representation import WassersteinDissimilarity
from .propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman
from .utilities import cv_classify, retrieve_graph_filenames, retrieve_binary_filenames, read_labels, \
    sparse_histogram, create_projection, create_metadata_file, cv_classify_automl
from .graphdata import Data
from .load import download_dataset
