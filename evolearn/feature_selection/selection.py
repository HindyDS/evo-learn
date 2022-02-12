import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional, Union

class RankSelection:
    '''Rank solution candidates according to fitness and select the top candidates.
    Rank Selection also works with negative fitness values and is mostly used when the individuals
    in the population have very close fitness values (this happens usually at the end of the run).
    This leads to each individual having an almost equal share of the
    pie (like in case of fitness proportionate selection) and hence each individual no matter
    how fit relative to each other has an approximately same probability of getting selected as a parent.
    This in turn leads to a loss in the selection pressure towards fitter individuals,
    causing the GA to make poor parent selections in such situations.

    Parameters
    ----------

    pct_survivors : int, float
    Argument that controls the number of survivors.
    '''
    def __init__(self, pct_survivors: Optional[Union[int, float]]):
        if pct_survivors > 1 or pct_survivors < 0:
            raise Exception("pct_survivors must be a fraction between 0 and 1.")

        self.pct_survivors = pct_survivors
        self.preselection_population_size = []
        self.survived_population_size = []

    def _select(self, population: pd.DataFrame) -> pd.DataFrame:
        """Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate."""
        self.preselection_population_size.append(len(population))

        # Sort and select top candidates according to their level of fitness
        population.sort_values('__fitness__', ascending=False, inplace=True)

        pct_survivors = int(len(population) * self.pct_survivors)

        self.survived_population_size.append(len(population))

        return population.head(pct_survivors)


class RouletteWheelSelection:
    '''In the roulette wheel selection, the probability of choosing an individual
    for breeding of the next generation is proportional to its fitness, the better the fitness is,
    the higher chance for that individual to be chosen. Choosing individuals can be depicted as
    spinning a roulette that has as many pockets as there are individuals in the current generation,
    with sizes depending on their probability.

    Parameters
    ----------
    population : {array-like, sparse matrix} of shape (n_samples, n_features)
    Fitness of each candidate.

    n_survivors : integer
    Argument that controls the number of survivors.
    '''

    def __init__(self, pct_survivors: Optional[Union[int, float]]):
        if pct_survivors > 1 or pct_survivors < 0:
            raise Exception("pct_survivors must be a fraction between 0 and 1.")

        self.pct_survivors = pct_survivors
        self.preselection_population_size = []
        self.survived_population_size = []

    def _select(self, population: pd.DataFrame) -> pd.DataFrame:
        """Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate."""

        pct_survivors: int
        prob: pd.Series
        selected_idx: int

        self.preselection_population_size.append(len(population))

        if len(population) == 0:
            raise Exception('Given population is empty for selection.')

        pct_survivors = int(len(population) * self.pct_survivors)

        # Calculate the fitness proportionate probabilities of selection according to candidates' fitness
        population.sort_values('__fitness__', inplace=True)
        # if all negatives, upside down the proba
        if sum(population['__fitness__'] < 0) == len(population):
            prob = population['__fitness__'] / sum(population['__fitness__'])
            prob = prob.iloc[::-1]

        elif sum(population['__fitness__'] >= 0) == len(population):
            prob = population['__fitness__'] / sum(population['__fitness__'])

        # Selecting (sampling) candidates from the distribution
        selected_idx = np.random.choice(a=population.index, size=pct_survivors, p=prob, replace=False)

        population = population.loc[selected_idx]

        self.survived_population_size.append(len(population))

        return population


class SteadyStateSelection:
    '''In every generation few are selected (good - with high fitness) chromosomes for creating a new offspring.
    Then some (bad - with low fitness) chromosomes are removed and the new offspring is placed in their place.
    The rest of population survives to new generation.

    Parameters
    ----------
    population : {array-like, sparse matrix} of shape (n_samples, n_features)
    Fitness of each candidate.
    '''

    def __init__(self, elimination_ratio: float=.3):
        if not 0 < elimination_ratio < 1:
            raise Exception('SteadyStateSelection argument "elimination_ratio" must be a float between 0 and 1.')

        self.elimination_ratio = elimination_ratio
        self.preselection_population_size = []
        self.survived_population_size = []

    def _select(self, population: pd.DataFrame) -> pd.DataFrame:
        """Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate."""

        elimination_ratio: Union[int, float]

        self.preselection_population_size.append(len(population))

        # Calculate the number of candidates being removed
        elimination_ratio = round(len(population) * self.elimination_ratio)

        # Remove bad candidates
        population = population.iloc[:-elimination_ratio].copy()

        self.survived_population_size.append(len(population))

        return population


class TournamentSelection:
    '''Tournament Selection is a Selection Strategy used for selecting the
    fittest candidates from the current generation in a Genetic Algorithm.
    These selected candidates are then passed on to the next generation.
    In a K-way tournament selection, we select k-individuals and run a
    tournament among them. Only the fittest candidate amongst those
    selected candidates is chosen and is passed on to the next generation.
    In this way many such tournaments take place and we have our final selection
    of candidates who move on to the next generation. It also has a parameter
    called the selection pressure which is a probabilistic measure of a candidate’s
    likelihood of participation in a tournament. If the tournament size is larger,
    weak candidates have a smaller chance of getting selected as it has to compete
    with a stronger candidate. The selection pressure parameter determines the rate
    of convergence of the GA. More the selection pressure more will be the Convergence rate.
    GAs are able to identify optimal or near-optimal solutions over a wide range of selection pressures.
    Tournament Selection also works for negative fitness values.

    Parameters
    ----------
    population : {array-like, sparse matrix} of shape (n_samples, n_features)
    Fitness of each candidate.

    k : integer (tournament_size)
    Argument that controls the number of participants in each tourament.
    '''

    def __init__(self, k: int=2, preserve_remainders: bool=True):
        if k < 0:
            raise Exception('TournamentSelection argument "k" must be a positive integer.')

        self.k = k
        self.preserve_remainders = preserve_remainders
        self.preselection_population_size = []
        self.survived_population_size = []

    def _select(self, population: pd.DataFrame) -> pd.DataFrame:
        """Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate."""

        participants: pd.DataFrame
        winners: list
        winner: pd.DataFrame

        self.preselection_population_size.append(len(population))

        winners = []
        while True:
            # Choose k (the tournament size) individuals from the population at random
            participants = population.sample(n=self.k).copy()
            population.drop(participants.index, inplace=True)

            # Select the fittest candidate
            participants.sort_values('__fitness__', inplace=True)
            winner = participants.head(1).copy()

            # Choose the best individual from the tournament
            winners.append(winner)

            # If preserve remainders, check if there are any candidates left for tournements
            if len(population) < self.k and self.preserve_remainders:
                winners.append(population)
                break

        population = pd.concat(winners)

        self.survived_population_size.append(len(population))

        return population


class StochasticUniversalSampling:
    '''SUS is a development of fitness proportionate selection(FPS)
    which exhibits no bias and minimal spread. Where FPS chooses several
    solutions from the population by repeated random sampling,
    SUS uses a single random value to sample all of the solutions by
    choosing them at evenly spaced intervals.This gives weaker members
    of the population(according to their fitness) a chance to be chosen.

    FPS can have bad performance when a member of the population has a
    really large fitness in comparison with other members.Using a comb-like
    ruler, SUS starts from a small random number, and chooses the next
    candidates from the rest of population remaining, not allowing the
    fittest members to saturate the candidate space.

    Parameters
    ----------
    population : {array-like, sparse matrix} of shape (n_samples, n_features)
    Fitness of each candidate.

    n_survivors : integer
    Argument that controls the number of survivors.
    '''
    def __init__(self, pct_survivors: Optional[Union[int, float]]):
        if not 0 < pct_survivors < 1:
            raise Exception("pct_survivors must be a fraction between 0 and 1.")

        self.pct_survivors = pct_survivors
        self.preselection_population_size = []
        self.survived_population_size = []

    def _select(self, population: pd.DataFrame) -> pd.DataFrame:
        """Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate."""

        pct_survivors: int
        prob: pd.Series
        starting_pointer: np.int64
        distance: int
        pointers: list
        remove_list: list

        self.preselection_population_size.append(len(population))

        if len(population) == 0:
            raise Exception('Given population is empty for selection.')

        pct_survivors = int(len(population) * self.pct_survivors)

        # Calculate the fitness proportionate probabilities of selection according to candidates' fitness
        population.sort_values('__fitness__', inplace=True)

        # if all negatives, upside donw the proba
        if (population['__fitness__'] < 0).any().any():
            prob = population['__fitness__'] / sum(population['__fitness__'])
            prob = prob.iloc[::-1]

        else:
            prob = population['__fitness__'] / sum(population['__fitness__'])

        # Initiate starting pointer according to probability
        starting_pointer = np.random.choice(range(len(population)), 1, p=prob, replace=False)[0]

        # Calculate the distance between each pointer
        distance = int(len(population) / pct_survivors)
        pointers = [starting_pointer]
        for i in range(1, pct_survivors):
            pointers.append(pointers[0] + 2 * i)

        # The wheel
        for i in pointers:
            if i - len(population) in pointers:
                break

            if i >= len(population):
                pointers = [i - len(population)] + pointers
                pointers.remove(i)

        remove_list = [i for i in pointers if i>= len(population)]
        for i in remove_list:
            pointers.remove(i)

        # Select candidates using the pointers
        population = population.iloc[pointers].copy()

        self.survived_population_size.append(len(population))

        return population


class BoltzmannSelection:
    '''In Boltzmann selection, a continuously varying temperature
    controls the rate of selection according to a preset schedule.
    The temperature starts out high, which means that the selection
    pressure is low. The temperature is gradually lowered, which
    gradually increases the selection pressure, thereby allowing the
    GA to narrow in more closely to the best part of the search space
    while maintaining the appropriate degree of diversity.

    Parameters
    ----------
    population : {array-like, sparse matrix} of shape (n_samples, n_features)
    Fitness of each candidate.

    n_survivors : integer
    Argument that controls the number of survivors.

    g : integer
    Current number of generation

    G : integer
    Maximum value of g

    a : float
    [0, 1]

    T0 : float
    [5, 100]

    f_max : float
    fitness of the currently available best string
    '''

    def __init__(self, pct_survivors: float, T0: Optional[Union[int, float]], a: Optional[Union[int, float]]) -> pd.DataFrame:
        if 0 < pct_survivors < 1:
            raise Exception("pct_survivors must be a fraction between 0 and 1.")

        self.pct_survivors = pct_survivors
        self.T0 = T0
        self.a = a
        self.preselection_population_size = []
        self.survived_population_size = []

    def _select(self, population: pd.DataFrame, f_max: Optional[Union[int, float]], g: int, G: int) -> pd.DataFrame:
        """Parameters
        ----------

        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate.

        f_max : int, float
        Best fitness achieved

        g : int
        Number of current generation

        G : int
        Maximum generation
        """

        f: int
        pct_survivors: int
        boltzmann_proba: pd.Series

        self.preselection_population_size.append(len(population))

        if len(population) == 0:
            raise Exception('Given population is empty for selection.')

        # If exsisting chromosomes do better, assign current best as the new best
        f = population['__fitness__'].max()
        if f > f_max:
            population.sort_values('__fitness__', ascending=False, inplace=True)
            pct_survivors = int(len(population) * self.pct_survivors)

            return population.iloc[:pct_survivors]

        # If no better solution found, then select chromosome with Boltzmann probabilities
        else:
            def get_boltzmann_proba(f):
                k = (1 + 100 * g / G)
                T = self.T0 * (1 - self.a) ** k
                boltzmann_proba = np.exp(-(f_max - f) / T)

                return boltzmann_proba

            # Calculate the Boltzmann probabilities of the candidates
            boltzmann_proba = population['__fitness__'].apply(lambda x: get_boltzmann_proba(x))

            # Normalizing the Boltzmann probabilities to avoid valueError (not sum up to 1)
            boltzmann_proba /= boltzmann_proba.sum()

            # if all negatives, upside down the proba
            if sum(population['__fitness__'] < 0) == len(population):
                boltzmann_proba = boltzmann_proba.iloc[::-1]

            # Sampling candidates from the distribution
            try:
                # Can be selected more than one time
                selected_idx = np.random.choice(a=range(len(population)), size=pct_survivors, p=boltzmann_proba, replace=False)

            except ValueError:
                selected_idx = np.random.choice(a=range(len(population)), size=pct_survivors, p=boltzmann_proba, replace=True)
                cuerrent_time = datetime.now().strftime("%H:%M:%S")
                warnings.warn(f'[{cuerrent_time}] Low Population Diversity Warning: Boltzmann Selection fail to select candidates without replacement. This might due to imbalanced Boltzmann probability distribution or desired child population size larger than parents population. This might lead to low population diversity. Please consider increasing the size of population or lowering the number of survivors.')

            population = population.iloc[selected_idx].copy()

            self.survived_population_size.append(len(population))

            return population
