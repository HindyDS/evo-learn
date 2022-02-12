import warnings
from datetime import datetime
from typing import Optional, Union
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


class MatingFunction:
    """Genetic optimization over hyper parameters.
    GenesSearchCV implements a "fit" method.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    GenesSearchCV will reproduce new parameter values in each generation. The
    evolution processes is mostly probabilistic.

    Parameters
    ----------
    pop_ratio : float, int
       Percentage of survived population.
       Determines how many couples are paired during mating.

    increst_prevention : boolean
       If True, solution candidates sharing the same parents will be paired together.

    Attributes
    ----------
    pop_ratio : float, int
        Percentage of survived population.
        Determines how many couples are paired during mating.

    increst_prevention : bool
        If True, solution candidates sharing the same parents will be paired together."""

    def __init__(self, pop_ratio: Union[float, int]=1, increst_prevention: bool=True):
        self.pop_ratio = pop_ratio
        self.increst_prevention = increst_prevention

    def _pair(self, population: pd.DataFrame) -> list:
        """Pairing solution candidates according to given parameters.

        Parameters
        ----------
        population : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        """

        n_couples: int
        current_time: str
        families: list
        n_families: int

        # n_n_couples defines how many couples that we want
        n_couples = int(self.pop_ratio * len(population))

        # group the population by parents
        families = list(population.groupby('__parents__'))
        n_families = len(families)
        family_parents = families[0][0]
        if n_families == 1:
            if (self.increst_prevention and family_parents == 'None') or not self.increst_prevention:
                def _couple(i) -> tuple:
                    family = families[0][1].drop(['__parents__', '__fitness__'], axis=1)
                    member_idxes = np.random.choice(a=range(len(family)), size=2, replace=False)
                    member1 = family.iloc[[member_idxes[0]]]
                    member2 = family.iloc[[member_idxes[1]]]

                    return (member1, member2)

            if self.increst_prevention and family_parents != 'None':
                warnings.warn('Cannot pair couples without increst.')
                return None

        # More than one family, pick two families without replacement
        elif n_families > 1:
            def _couple(i) -> tuple:
                couples = []
                # Randomly pick two distinct families
                families_idxes = np.random.choice(a=range(len(families)), size=2, replace=False)

                # Randomly pick one member from family one
                family1 = families[families_idxes[0]][1].drop(['__parents__', '__fitness__'], axis=1)
                family_idx1 = np.random.choice(a=range(len(family1)))
                member1 = family1.iloc[[family_idx1]]

                # Randomly pick one member from family two
                family2 = families[families_idxes[1]][1].drop(['__parents__', '__fitness__'], axis=1)
                family_idx2 = np.random.choice(a=range(len(family2)))
                member2 = family2.iloc[[family_idx2]]

                return (member1, member2)


        # Run _couple function in parallel
        parents = Parallel(n_jobs=-1)(delayed(_couple)(i) for i in range(n_couples))

        # If the amount of couples paired is smaller than desired amount, then warn
        if len(parents) < n_couples:
            current_time = datetime.now().strftime("%H:%M:%S")
            warnings.warn(f'[{current_time}] Population Decline Warning: Cannot pair enough couples during mating.')

        return parents

