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

    def __init__(self, k: int, c_pt: Union[int, str]='random'):
        self.k = k
        self.c_pt = c_pt

    def _crossover(self, population: list, k: int, c_pt: Union[int, str], gen:int) -> tuple:
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

        if isinstance(population, type(None)):
            return None

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


class LinearCombinationCrossover:
    """Constructing child population by multiplying genes of
    the parents by a constant and adding the results

    Parameters
    ----------
    a : float
    Constant term that scale down the values
    of genes of the child population.
    """

    def __init__(self, a: float):
        if not 0 < a < .5:
            raise Exception('LinearCombinationCrossover argument a must be a fraction between 0 and 0.5.')

        self.a = a

    def _reproduce(self, parents: list, gen:int) -> list:
        """Parameters
        ----------

        parents : float
        List of parents used to reproduce child population.

        gen : int
        Denotes the current number of generation."""
        if isinstance(population, type(None)):
            return None

        def _sub_linear_combination(parents, a):
            # Store the datatype of the genes
            genes_dtype = list(parents[0].drop('__fitness__', axis=1).dtypes.values) + [str]

            chromosome1 = parents[0].drop('__fitness__', axis=1).copy()
            chromosome2 = parents[1].drop('__fitness__', axis=1).copy()
            child_columns = parents[0].columns
            child1 = chromosome1.iloc[0] + abs(chromosome2.iloc[0] - chromosome1.iloc[0])
            child2 = chromosome2.iloc[0] + abs(chromosome1.iloc[0] - chromosome2.iloc[0])
            child1 = child1.to_frame().T
            child2 = child2.to_frame().T
            child1['__parents__'] = f"{gen}-{chromosome1.index[0]}-{chromosome2.index[0]}"
            child2['__parents__'] = f"{gen}-{chromosome1.index[0]}-{chromosome2.index[0]}"

            # Preserve the datatypes
            for g, t in zip(list(child1.columns) + ['__parents__'], genes_dtype):
                child1[g] = child1[g].astype(t)
                child2[g] = child2[g].astype(t)

            return (child1, child2)


        # For each pairs, reproduce offsprings
        children = Parallel(n_jobs=-1)(delayed(_sub_linear_combination)(p, self.a) for p in parents)

        # For each pairs, reproduce offsprings
        children = pd.concat(c for chromosomes in children for c in chromosomes)

        return children


class FitnessProportionateAverage:
    """Fitness proportional average will be calculated as weights
    to reproduce child population."""

    def __init__(self):
        cuerrent_time = datetime.now().strftime("%H:%M:%S")
        warnings.warn(f'[{cuerrent_time}] Population Decline Warning: The size of the child population reproduced by FitnessProportionateAverage will be half of the parents population. This might lead to early stopping.')

    def _reproduce(self, parents, gen):
        '''
        Parameters
        ----------

        parents : float
        List of parents used to reproduce child population.

        gen : int
        Denotes the current number of generation.
        '''

        if isinstance(population, type(None)):
            return None

        def _sub_reproduce(p):
            chromosome1 = p[0]
            chromosome2 = p[1]

            if 'proba' in chromosome1.columns:
                chromosome1.drop('proba', axis=1, inplace=True)

            if 'proba' in chromosome2.columns:
                chromosome2.drop('proba', axis=1, inplace=True)

            # Store the datatype of the genes
            genes_dtype = list(chromosome1.dtypes.values) + [str]

            # Calulcate the total fitness
            total_fitneess = chromosome1['fitness'].values[0] + chromosome2['fitness'].values[0]

            # Calulcate the fitness proportionate probabilties
            chromosome1['proba'] = chromosome1['fitness'] / total_fitneess
            chromosome2['proba'] = chromosome2['fitness'] / total_fitneess

            # Calulcate the fitness proportionate weighted average
            x1 = chromosome1.values[0]
            x2 = chromosome2.values[0]
            w1 = chromosome1['proba'].values[0]
            w2 = chromosome2['proba'].values[0]

            child1 = (x1 * w1 + x2 * w2) / (w1 + w2)
            child2 = (x1 * w2 + x2 * w1) / (w1 + w2)

            child1 = pd.DataFrame(child1, index=chromosome1.columns).T
            child2 = pd.DataFrame(child2, index=chromosome1.columns).T

            child1['parents'] = f"{gen}-{chromosome1.index[0]}-{chromosome2.index[0]}"
            child2['parents'] = f"{gen}-{chromosome1.index[0]}-{chromosome2.index[0]}"

            # Preserve the datatypes
            for g, t in zip(chromosome1.columns, genes_dtype):
                child1[g] = child1[g].astype(t)
                child2[g] = child2[g].astype(t)

            if 'fitness' in child1.columns:
                child1.drop(['fitness'], axis=1, inplace=True)
                child2.drop(['fitness'], axis=1, inplace=True)

            if 'proba' in child1.columns:
                child1.drop(['proba'], axis=1, inplace=True)
                child2.drop(['proba'], axis=1, inplace=True)

            return child1, child2

        # For each pairs, reproduce offsprings
        children = Parallel(n_jobs=-1)(delayed(_sub_reproduce)(p) for p in parents)

        # For each pairs, reproduce offsprings
        children = pd.concat([c for chromosomes in children for c in chromosomes])

        return children
