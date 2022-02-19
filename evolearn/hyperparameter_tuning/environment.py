import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Optional, Union

class AdaptiveReproduction:
    """Adaptively change the mating rate to control the overall
    population size at certian level

    Parameters
    ----------
    pop_cap : int, default : None
    Maximum population size
    """
    def __init__(self, pop_cap: Optional[Union[int]]=None):
        self.current_pop_size = None
        self.pop_cap = pop_cap

    def _control(self, population: Optional[Union[list, pd.DataFrame]]) -> Optional[Union[bool, pd.DataFrame]]:
        '''Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
            Solution candidates
        '''

        current_time: str

        # Check if the current population is declining, if so, warning
        current_time = datetime.now().strftime("%H:%M:%S")
        if self.current_pop_size == None:
            self.current_pop_size = len(population)

        else:
            if self.current_pop_size < len(population):
                warnings.warn(f'[{current_time}] Population Decline Warning: The current population size is smaller than previous. This might lead to premature convergence.')

        # Adaptive control
        if isinstance(self.pop_cap, int):
            return (self.pop_cap / len(population)) * 1.86

        else:
            # Early Stopping when there are no candidates, if True, will break process
            if population == [] and self.pop_cap == None:
                warnings.warn(f'[{current_time}] Population Decline Warning: No couples are paired during the mating procees. Please consider increase the initial population size and increase the number of survivors of the selection process.')
                return True

        return False


class AdaptiveMutation:
    """Adaptively change the mutation rate to control the overall
    population diversity.

    Parameters
    ----------
    a : int, float, default : .2
    Adjustment factor of self-adaptive mutation rate.
    """

    def __init__(self, a: Optional[Union[int, float]]=.2):
        self.a = a

    def _adaptive_mutation_rate(self, population: pd.DataFrame) -> Optional[Union[int, float]]:
        '''Self-adaptive mutation rate (Maximum percentage * alpha).

            Parameters
            ----------
            population : {array-like, sparse matrix} of shape (n_samples, n_features)
            Solution candidates
        '''

        total_len: int
        pct_dup: pd.Series

        population = pd.DataFrame(population).astype(str)
        total_len = len(population)
        pct_dup = population.groupby(population.columns.tolist(), as_index=False).size()['size'] / total_len

        return pct_dup.max() * self.a


class Elitism:
    '''Often to get better parameters, strategies with partial experimental are used.
    One of them is elitism, in which a small portion of the best individuals from the
    last generation is carried over (without any changes) to the next one.

    Parameters
    ----------
    pct : int, float, default : .05
    Percentage of population being selected as elites.
    '''

    def __init__(self, pct: Optional[Union[int, float]]=.05):
        if not (0 < pct <=1):
            raise Exception('ElitismControl argument pct must a positive integer smaller than one.')

        self.pct = pct
        self.elite_and_pop = None

    def _preserve_elitism(self, population: pd.DataFrame) -> pd.DataFrame:
        '''
        Parameters
        ----------
        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Solution candidates
        '''

        pct: int

        population.sort_values('__fitness__', ascending=False, inplace=True)

        # If no elites are selected due to small population, then warn
        pct = int(len(population) * self.pct)
        if pct == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            warnings.warn(
                f'[{current_time}] Elitism Failed Warning: Elites selected from this generation is 0. Number of elite is automatically set to 1. Please consider increase "pct".')
            pct = 1

        # If self.elite_and_pop is None, set top candidates as elite_and_pop
        if isinstance(self.elite_and_pop, type(None)):
            self.elite_and_pop = population.head(pct)
            return pd.concat([self.elite_and_pop, population]).drop_duplicates()

        # If self.elite_and_pop is not none, concat current population with self.elite_and_pop
        else:
            self.elite_and_pop = pd.concat([self.elite_and_pop, population.head(pct)])
            self.elite_and_pop = self.elite_and_pop.sort_values('__fitness__', ascending=False).head(pct)

            return pd.concat([self.elite_and_pop, population]).drop_duplicates()

    def _drop_elite(self, population: pd.DataFrame) -> pd.DataFrame:
        """Remove selected elites

        Parameters
        ----------
        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Solution candidates

        """
        for i in range(len(self.elite_and_pop)):
            elite = self.elite_and_pop.iloc[[i]]
            elite_select = population.isin(elite).drop('__parents__', axis=1).sum(axis=1) == len(population.columns) - 1
            population = population[np.where(elite_select == False, True, False)].copy()

        return population
