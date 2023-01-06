from wwlr.utilities import cv_classify_kernel
import numpy as np
import argparse
import os
import pickle
import pandas as pd

from shutil import copy2
from datetime import datetime
from tensorflow import summary
from tensorboard.plugins.hparams import api as hp

import warnings
warnings.filterwarnings("ignore")


from wwlr import WeisfeilerLehman, ContinuousWeisfeilerLehman, WassersteinDissimilarity, \
     Data, cv_classify, create_projection, create_metadata_file, download_dataset, \
    cv_classify_automl


VERSION = '1.1'

#Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name')
    parser.add_argument('-r', '--references', type=int, help="(Max) number of reference networks")
    parser.add_argument('-i', '--iterations', type=int, help="(Max) number of WL iterations")

    parser.add_argument('--categorical', default=False, action='store_true', help='Use categorical WL scheme')
    parser.add_argument('--gridsearch', default=False, action='store_true', help='Enable grid search')
    parser.add_argument('--bipartite', default=False, action='store_true', help='Treat bipartite sets separately')
    parser.add_argument('--output', default=None, type=str, help='Optional output folder name')
    parser.add_argument('--random_refs', default=False, action='store_true',
                        help='Use random networks as references, in stead of cluster mediods')
    parser.add_argument('--gexf', default=False, action='store_true', help='Data in networkx gexf format')
    parser.add_argument('--online', default=False, action='store_true', help="Download dataset")
    parser.add_argument('--automl', default=False, action='store_true', help='Use auto-sklearn')
    parser.add_argument('--model', type=str, default="wwlr", help='Select model: wwlr, wwl')
    parser.add_argument('--representations', type=str, default=None, help='Use representations of this folder')
    parser.add_argument('--subset', type=float, default=None, help='random subset of graphs to speed up calculations (online only!)')

    parser.add_argument('--n_jobs', type=int, default=1)
    #---------------------------------------------
    # Parse arguments: general vs model specific
    #---------------------------------------------
    args = parser.parse_args()
    dataset = args.dataset
    dataset_short_name = os.path.splitext(os.path.basename(dataset))[0]
    print(args)

    print(f'Generating results for {dataset_short_name}...')

    #-------------------------------------------
    # Create output folder for experiment
    #------------------------------------------
    output_path = os.path.join('Tensorboards', dataset_short_name)
    data_path = os.path.join('data', dataset_short_name)

    # create metadata dir
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if args.representations:
        representations_path = os.path.join(output_path, args.representations)

    if args.output:
        output_path = os.path.join(output_path, args.output)
    else:
        output_path = os.path.join(output_path, datetime.now().strftime("%Y-%m-%d_%H-%M"))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, 'arguments.txt'), 'w') as f:
        f.write(f'Version: {VERSION}')
        for k, v in vars(args).items():
            f.write(f'\n{str(k)}: {str(v)}')

    #-------------------------------------------
    # Load datasets
    #-------------------------------------------
    data = Data()

    if args.online:
        # Download dataset
        print(f'Downloading dataset: {dataset}')

        graphdict = download_dataset(dataset, args.categorical, args.subset)
        data.graphs = list(graphdict.values())
        data.names = list(graphdict.keys())
        data.labels = pd.DataFrame({'category': np.array([v.graph['label'] for v in data.graphs])})
    else:
        data_path = os.path.join('data', dataset)
        y = pd.read_csv(os.path.join(data_path, 'labels.tsv'), sep='\t', dtype='category')
        for col in y:
            # category -> numerical value
            y[col] = y[col].cat.codes

        data.create_graph_generator(data_path, args.gexf)
        data.labels = y

    if args.bipartite and not args.representations:
        bss = [np.array([e for e, (_, v) in enumerate(g.nodes('bipartite')) if v]) for g in data.graphs]
        # The graph generator has been used, so it needs to be recreated
        data.create_graph_generator(data_path, args.gexf)
    else:
        bss = []


    if args.gridsearch:
        iters = list(range(args.iterations))
        HP_ITERATIONS = hp.HParam('Iterations', hp.Discrete(iters))

        # Trying every possible r up to args.references would lead to too many representations to classify
        refs_log2 = np.log2(args.references)
        refs = np.logspace(1, refs_log2, num=round(refs_log2), base=2, dtype=int).tolist()
        HP_REFERENCES = hp.HParam('References', hp.Discrete([np.inf] if args.model == 'wwl' else refs))
        HP_CLASSIFIER = hp.HParam('Classifier', hp.Discrete(['automl', 'svm'] if args.automl else ['svm']))
    else:
        HP_ITERATIONS = hp.HParam('Iterations', hp.Discrete([args.iterations]))
        HP_REFERENCES = hp.HParam('References', hp.Discrete([np.inf] if args.model == 'wwl' else [args.references]))
        HP_CLASSIFIER = hp.HParam('Classifier', hp.Discrete(['automl' if args.automl else 'svm']))

    metrics = []
    for e, col in enumerate(data.labels.columns):
        metrics.append(hp.Metric(f'accuracy-{e}-mean', display_name=f'Accuracy {col} (mean)'))
        metrics.append(hp.Metric(f'accuracy-{e}-std', display_name=f'Accuracy {col} (std)'))

    with summary.create_file_writer(output_path).as_default():
        hp.hparams_config(
            hparams=[HP_ITERATIONS, HP_REFERENCES, HP_CLASSIFIER],
            metrics=metrics,
        )

    # ---------------------------------
    # Embeddings
    # ---------------------------------
    out_name = 'wl_embeddings.pickle'

    if not args.representations:
        try:
            with open(os.path.join(output_path, out_name), 'rb') as handle:
                label_sequences = pickle.load(handle)
        except Exception:
            print(f'Generating {"categorical" if args.categorical else "continuous"} embeddings for {dataset}.')

            if args.categorical:
                ps = WeisfeilerLehman(args.iterations)
            else:
                ps = ContinuousWeisfeilerLehman(args.iterations)
            label_sequences = ps.fit_transform(data.graphs)

            # Save embeddings to output folder
            with open(os.path.join(output_path, out_name), 'wb') as handle:
                pickle.dump(label_sequences, handle)
            print(f'Embeddings for {dataset} computed, saved to {os.path.join(output_path, out_name)}.')

    # ---------------------------------
    # Wasserstein computations & classification
    # ---------------------------------
    print('Computing the Wasserstein distances and classifying...')

    for i in HP_ITERATIONS.domain.values:
        dr = WassersteinDissimilarity(n_refs=1, categorical=args.categorical, random_refs=args.random_refs, n_jobs=args.n_jobs)
        # Call fit to generate dr.all_distances

        if args.representations:
            representations_run_path = os.path.join(representations_path, f'i{i}-r{HP_REFERENCES.domain.values[0]}')
            with open(os.path.join(representations_run_path, 'all_distances.pickle'), 'rb') as handle:
                all_distances = pickle.load(handle)
            dr.all_distances = all_distances
        else:
            n_features = label_sequences[0].shape[1] // (args.iterations + 1)
            sequences = [s[:, :(i+1)*n_features] for s in label_sequences]
            dr.fit(sequences, bipartite_sets=bss, emd_max_iter=2000000, hist_max_iter=100)

        for r in HP_REFERENCES.domain.values:
            run_path = os.path.join(output_path, f'i{i}-r{r}')
            if not os.path.exists(run_path):
                os.mkdir(run_path)

            for c in HP_CLASSIFIER.domain.values:
                subrun_path = os.path.join(run_path, f'{c}')
                if not os.path.exists(subrun_path):
                    os.mkdir(subrun_path)

                dr.n_refs = r

                if c == 'automl':
                    best_means, best_stds = cv_classify_automl(dr, data.labels.values)
                elif args.model == 'wwl':
                    best_means, best_stds = cv_classify_kernel(dr, data.labels.values, n_jobs=args.n_jobs)
                else:
                    best_means, best_stds = cv_classify(dr, data.labels.values, n_jobs=args.n_jobs)

                print(f'Iterations: {i}, references: {r}, classifier: {c}')
                print(f'Means: {[round(m, 4) for m in best_means]}')
                print(f'STDs: {[round(s, 4) for s in best_stds]}')

                with summary.create_file_writer(subrun_path).as_default():
                    hp.hparams({
                        HP_ITERATIONS: i,
                        HP_REFERENCES: r,
                        HP_CLASSIFIER: c
                    })
                    for e, mean in enumerate(best_means):
                        summary.scalar(f'accuracy-{e}-mean', mean, step=1)
                    for e, std in enumerate(best_stds):
                        summary.scalar(f'accuracy-{e}-std', std, step=1)

            metadata_file = os.path.join(data_path, 'metadata.tsv')
            if os.path.isfile(metadata_file):
                copy2(metadata_file, run_path)
            else:
                create_metadata_file(data.names, data.labels, run_path)

            representations = dr.fit_transform_subset(np.arange(len(dr.all_distances)))
            create_projection(representations, run_path)

            with open(os.path.join(run_path, 'all_distances.pickle'), 'wb') as handle:
                pickle.dump(dr.all_distances, handle)

            with open(os.path.join(run_path, 'representations.pickle'), 'wb') as handle:
                pickle.dump(representations, handle)


if __name__ == '__main__':
    main()
