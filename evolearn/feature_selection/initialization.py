import numpy as np
import pandas as pd
from typing import Optional, Union


class Genes:
    """Initialize a population to selection

    Parameters
    ----------
    heuristic : list
    Where it contains the pre-selected features.
    If not None, the initial population will carry
    the genes from the list.

    pop_size : int
    Size of the initial population."""

    def __init__(self, pop_size: int, heuristic: Optional[Union[list, None]]=None):
        self.pop_size = pop_size
        self.heuristic = heuristic

    def _populate(self, X) -> pd.DataFrame:
        """
        Parameters
        ----------

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.
        """
        population: Union[list, pd.DataFrame]

        population = [np.random.choice(a=[True, False], size=len(X.columns)) for i in range(self.pop_size)]
        population = pd.DataFrame(population, columns=X.columns)
        population = population[population.sum(axis=1) != 0].drop_duplicates()  # not enough genes

        if self.heuristic != None:
            for h in self.heuristic:
                population.loc[:, h] = True

        population['__parents__'] = 'None'
        population['__fitness__'] = 'None'

        return population