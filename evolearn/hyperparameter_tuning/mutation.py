import numpy as np
import pandas as pd
from typing import Optional, Union

class Boundary:
    """Randomly choose and set the values of a small
    amount genes to the upper or lower limit

    Parameters
    ----------
    epsilon : float
    Mutation rate that determines if genes will mutate or not.
    """

    def __init__(self, epsilon: Optional[Union[int, float]]=.15):

        self.epsilon = epsilon

    def _mutate(self, offspring: pd.DataFrame) -> pd.DataFrame:
        """Parameters
        ----------
        offspring : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate.
        """

        genes: str
        upper: Union[int, float]
        lower: Union[int, float]
        mutagen: Union[int, float]
        mutate_idx: Union[int, float]

        for genes in offspring.drop('__parents__', axis=1).columns:
            if self.epsilon > np.random.random():
                upper = max(offspring[genes])
                lower = min(offspring[genes])
                mutagen = np.random.choice(a=[lower, upper])[0]
                mutate_idx = np.random.choice(a=range(len(offspring)))[0]
                offspring.loc[:, genes].iloc[mutate_idx] = mutagen

        return offspring

class Shrink:
    def __init__(self, epsilon: Optional[Union[int, float]]=.15, prior: str='normal') -> pd.DataFrame:
        '''This operator adds a random number taken from a Gaussian distribution
        with mean equal to the original value of each decision variable
        characterizing the entry parent vector.

        Parameters
        ----------

        epsilon : float [0, 1]
        Mutation probability of a chromosome
        '''

        self.epsilon = epsilon
        self.prior = prior

    def _mutate(self, offspring: pd.DataFrame) -> pd.DataFrame:
        """Parameters
        ----------
        offspring : {array-like, sparse matrix} of shape (n_samples, n_features)
        Chromosomes in pandas dataframe
        """

        genes: str
        mu: float
        sigma: float
        mu_idx: float
        sigma_idx: float
        mutagen: Union[int, float]
        mutate_idx: Union[int, float]
        mutating_gene: Union[np.int64, np.float64]

        # Store the datatype of the genes
        genes_dtype = offspring.dtypes.values

        # For each gene, check if mutation occur
        # only if gene is not categorical genes.
        for genes in offspring.drop('__parents__', axis=1).columns:
            if self.epsilon > np.random.random():
                if not isinstance(offspring[genes].sample(n=1).iloc[0], str):
                    mu = offspring[genes].mean()
                    sigma = offspring[genes].std()
                    mu_idx = np.array(range(len(offspring))).mean()
                    sigma_idx = np.array(range(len(offspring))).std()

                    while True:
                        # Mutate a single gene (a single cell) with a normally sampled value
                        if self.prior == 'normal' or self.prior == 'gaussian':
                            mutagen = np.random.normal(loc=mu, scale=sigma)
                            mutate_idx = np.random.normal(loc=mu_idx, scale=sigma_idx)
                            mutate_idx = int(mutate_idx)

                        elif self.prior == 'uniform':
                            mutagen = np.random.uniform(loc=mu, scale=sigma)
                            mutate_idx = np.random.uniform(loc=mu_idx, scale=sigma_idx)
                            mutate_idx = int(mutate_idx)

                        if mutate_idx <= len(offspring):
                            break

                    mutating_gene = offspring.loc[:, genes].iloc[mutate_idx]
                    if mutagen < 0:
                        mutagen = abs(mutagen)

                    if isinstance(mutating_gene, np.int64):
                        mutagen = int(mutagen)

                    mutating_gene = mutagen

        for g, t in zip(offspring.columns, genes_dtype):
            offspring[g] = offspring[g].astype(t)

        return offspring