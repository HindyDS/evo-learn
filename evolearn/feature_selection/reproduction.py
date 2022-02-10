import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from joblib import Parallel, delayed
from typing import Optional, Union

class KPointCrossover:
    """Splitting a pair of chromosomes into two segments and recombine
    with each other. Where 'K' denote how times will the chromosomes be splitted.

    Parameters
    ----------
    k : int
    Number of times of the chromosomes being splitted.

    c_pt: int, str
    If int, c_pt will be the position index of the splitting points.
    If str, the splitting point location where be randomly determined.
    """

    def __init__(self, k: Union[int, str], c_pt: Union[int, str]='random'):
        self.k = k
        self.c_pt = c_pt

    def _crossover(self, population: tuple, k: int, c_pt: Union[int, str], gen:int) -> tuple:
        """Parameters
        ----------
        population : list
        A list of paired chromosomes.

        gen: int
        Denotes which gen currently runing. Also, used to name the parents for the child population.
        """

        chrom1: pd.DataFrame
        chrom2: pd.DataFrame
        chrom1_segment1: pd.Series
        chrom1_segment2: pd.Series
        chrom2_segment1: pd.Series
        chrom2_segment2: pd.Series
        child1: Union[np.array, pd.DataFrame]
        child2: Union[np.array, pd.DataFrame]
        parents: str
        children: pd.DataFrame

        def _sub_crossover(p, c_pt):
            chrom1 = p[0]
            chrom2 = p[1]
            parents = [f"{gen}-{chrom1.index[0]}-{chrom2.index[0]}"]
            for i in range(self.k):
                if c_pt == 'random':
                    c_pt = np.random.choice(a=range(1, len(chrom1.columns)))

                chrom1_segment1 = chrom1.iloc[0, :c_pt].copy()
                chrom1_segment2 = chrom1.iloc[0, c_pt:].copy()
                chrom2_segment1 = chrom2.iloc[0, :c_pt].copy()
                chrom2_segment2 = chrom2.iloc[0, c_pt:].copy()

                # Use the obtained values to reproduce childrn
                child1 = np.hstack([chrom1_segment1, chrom2_segment2])
                child2 = np.hstack([chrom2_segment1, chrom1_segment2])

                # Make sure the returned children share the exact same datatype with thier parents
                child1 = pd.DataFrame(child1, index=chrom1.columns, columns=parents).T
                child2 = pd.DataFrame(child2, index=chrom2.columns, columns=parents).T

                chrom1 = child1
                chrom2 = child2

            # Define parents
            child1['__parents__'] = parents
            child2['__parents__'] = parents

            child1.reset_index(drop=True, inplace=True)
            child2.reset_index(drop=True, inplace=True)

            return (child1, child2)

        offspring = Parallel(n_jobs=-1)(delayed(_sub_crossover)(p, c_pt) for p in population )

        return offspring

    def _reproduce(self, population: list, gen: int) -> pd.DataFrame:
        """Parameters
        ----------
        population : list
        A list of paired chromosomes.

        gen: int
        Denotes which gen currently runing.
        Also, used to name the parents for the child population.
        """

        population: Union[list, pd.DataFrame]

        population = self._crossover(population=population, k=self.k, c_pt=self.c_pt, gen=gen)

        # For each pairs, reproduce offsprings
        population = [c for chromosomes in population for c in chromosomes]

        population = pd.concat(population)

        return population
