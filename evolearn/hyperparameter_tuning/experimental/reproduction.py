import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from joblib import Parallel, delayed

class AvergeReproduction:
    def __init__(self):
        cuerrent_time = datetime.now().strftime("%H:%M:%S")
        warnings.warn(f'[{cuerrent_time}] Population Decline Warning: The size of the child population reproduced by AvergeReproduction will be half of the parents population. This might lead to early stopping.')

    def _reproduce(self, parents, gen):
        '''
        A new child will be reproduced by calculating the average of the parents.

        Parameters
        ----------
        chromosome1 : {array-like, sparse matrix} of shape (1, n_features)
        Single solution candidate as parent.

        chromosome2 : {array-like, sparse matrix} of shape (1, n_features)
        Single solution candidate as parent.
        '''

        def _sub_reproduce(p):
            chromosome1 = p[0]
            chromosome2 = p[1]

            if 'fitness' in chromosome1.columns:
                chromosome1.drop('fitness', axis=1, inplace=True)

            if 'fitness' in chromosome2.columns:
                chromosome2.drop('fitness', axis=1, inplace=True)

            # Store the datatype of the genes
            genes_dtype = list(chromosome1.dtypes.values) + [str]

            # Reproducing new child by calculating the average
            child = pd.DataFrame(
                (chromosome1.values[0] + chromosome2.values[0]) / 2,
                index=chromosome1.columns
            ).T

            child['parents'] = f"{gen}-{chromosome1.index[0]}-{chromosome2.index[0]}"

            # Preserve the datatypes
            for g, t in zip(chromosome1.columns, genes_dtype):
                if t in ['int64', 'int32']:
                    child[g] = int(round(child[g]))

            child[g] = child[g].astype(t)

            return child

        # Store the datatype of the genes
        genes_dtype = list(parents[0][0].dtypes.values) + [str]

        # For each pairs, reproduce offsprings
        children = pd.concat(Parallel(n_jobs=-1)(delayed(_sub_reproduce)(p) for p in parents))

        return children
