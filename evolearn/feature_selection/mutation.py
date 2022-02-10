import pandas as pd
import numpy as np


class BitStringMutation:
    """Bit flipping a gene if selected

    Parameters
    ----------
    epsilon : float
    Mutation rate that determines if genes will mutate or not.
    """
    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon

    def _mutate(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate.
        """

        for c in range(len(population)):
            chromo = population.iloc[c].drop('__parents__')
            for g in range(len(chromo)):
                if self.epsilon > np.random.random():
                    if chromo.iloc[g] == True:
                        chromo.iloc[g] = False
                    else:
                        chromo.iloc[g] = True

        return population


class ExchangeMutation:
    """A pair of genes will be selected and exchange
    position with each other.

    Parameters
    ----------
    epsilon : float
    Mutation rate that determines if genes will mutate or not.
    """
    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon

    def _mutate(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate.
        """

        mutated = []
        for c in range(len(population)):
            chromo = population.iloc[c].drop('__parents__')
            parents = population.iloc[c].loc['__parents__']
            if self.epsilon > np.random.random():
                chromo_idx = np.random.choice(a=range(len(chromo)), size=2)
                chromo.iloc[chromo_idx[0]], chromo.iloc[chromo_idx[1]] = chromo.iloc[chromo_idx[1]], chromo.iloc[chromo_idx[0]]

            chromo = pd.DataFrame(chromo, index=chromo.index).T
            chromo['__parents__'] = parents
            mutated.append(chromo)

        return pd.concat(mutated)


class ShiftMutation:
    """A gene will be moved from one position to another.
    The adjacent genes will also be shifted together as well.

    Parameters
    ----------
    epsilon : float
    Mutation rate that determines if genes will mutate or not.
    """

    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon

    def _mutate(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate.
        """

        population_no_mutation = population.copy()
        shifted = []
        for c in range(len(population)):
            chromo = population.iloc[c].drop('__parents__')
            mut = chromo.to_list()
            if self.epsilon > np.random.random():
                pos = np.random.choice(a=range(len(mut)), size=2)
                g1 = mut[pos[0]]
                dir = int(pos[1] - pos[0])
                if dir != 0:
                    for i in range(pos[0], pos[1], dir):
                        mut[i] = mut[i + dir]
                        mut[pos[1]] = g1

            mut = pd.DataFrame(mut, index=chromo.index).T
            shifted.append(mut)

        if shifted == []:
            return population_no_mutation

        shifted = pd.concat(shifted)

        return shifted