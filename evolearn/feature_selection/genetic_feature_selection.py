import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

from evolearn.feature_selection.initialization import Genes
from evolearn.feature_selection.evaluation import FitnessFunction
from evolearn.feature_selection.selection import (RankSelection,
                                        RouletteWheelSelection,
                                        SteadyStateSelection,
                                        TournamentSelection,
                                        StochasticUniversalSampling,
                                        BoltzmannSelection
                                        )
from evolearn.feature_selection.mating import MatingFunction
from evolearn.optimization.reproduction import KPointCrossover
from evolearn.feature_selection.mutation import (BitStringMutation,
                                        ExchangeMutation,
                                        ShiftMutation)

class GeneticFeatureSelection:
    """Genetic feature selection over a set of features.
    GenesSearchCV implements a "fit" method.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    GeneticFeatureSelection will reproduce new parameter values in each generation. The
    evolution processes is mostly probabilistic.

    Parameters
    ----------
    n_gen : int
       Maximum number of generation (or loop) GenesSearchCV will run.

    initialization_fn : optimization.initialization.Genes class
       Class object to generate solution candidates.

    fitness_fn : optimization.evaluation.FitnessFunction
       Class object to evalute the fitness of solution candidates.

    selection_fn : can either be
       - optimization.selection.RankSelection,
       - optimization.selection.RouletteWheelSelection,
       - optimization.selection.SteadyStateSelection,
       - optimization.selection.TournamentSelection,
       - optimization.selection.StochasticUniversalSampling,
       - optimization.selection.BoltzmannSelection

       Class object to evalute the fitness of solution candidates.

    mating_fn : optimization.mating.MatingFunction
       Class object to pair the solution candidates for reproduction.

    reproduction_fn : can either be
       - optimization.reproduction.KPointCrossover,
       - optimization.reproduction.LinearCombinationCrossover,
       - optimization.reproduction.FitnessProportionateAverage

       Class object to reproduce child population.

    mutation_fn : can either be
       - optimization.mutation.Boundary,
       - optimization.mutation.Shrink

       Class object to mutate the child population.

    adaptive_population : optimization.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mating rate of the mating_fn.

    elitism : optimization.environment.Elitism, default: None
       Class object to perform elites selection, ace comparison and elites' traits induction.

    adaptive_mutation : optimization.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mutation probaility of the mutation_fn.

    Attributes
    ----------
    n_gen : int
       Maximum number of generation (or loop) GenesSearchCV will run.

    initialization_fn : optimization.initialization.Genes class
       Class object to generate solution candidates.

    fitness_fn : optimization.evaluation.FitnessFunction
       Class object to evalute the fitness of solution candidates.

    selection_fn : can either be
       - optimization.selection.RankSelection,
       - optimization.selection.RouletteWheelSelection,
       - optimization.selection.SteadyStateSelection,
       - optimization.selection.TournamentSelection,
       - optimization.selection.StochasticUniversalSampling,
       - optimization.selection.BoltzmannSelection

       Class object to evalute the fitness of solution candidates.

    mating_fn : optimization.mating.MatingFunction
       Class object to pair the solution candidates for reproduction.

    reproduction_fn : can either be
       - optimization.reproduction.KPointCrossover,
       - optimization.reproduction.LinearCombinationCrossover,
       - optimization.reproduction.FitnessProportionateAverage

       Class object to reproduce child population.

    mutation_fn : can either be
       - optimization.mutation.Boundary,
       - optimization.mutation.Shrink

       Class object to mutate the child population.

    adaptive_population : optimization.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mating rate of the mating_fn.

    elitism : optimization.environment.Elitism, default: None
       Class object to perform elites selection, ace comparison and elites' traits induction.

    adaptive_mutation : optimization.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mutation probaility of the mutation_fn.

    best_params_ : dict
       Best hyperparamters found

    best_fitness_ : float
       Best fitness achieved

    max_fitness_records_ : list
       Best fitness achieved of each generation

    mean_fitness_records_ : list
       Average fitness achieved of each generation

    preselection_population_size_ : list
       Population size of each generation before selection

    survived_population_size : list
       Population size of each generation after selection"""

    def __init__(self, n_gen:int, initialization_fn, fitness_fn, selection_fn, mating_fn, reproduction_fn, mutation_fn, adaptive_population=None, elitism=None, adaptive_mutation=None):
        self.n_gen = n_gen
        self.initialization_fn = initialization_fn
        self.fitness_fn = fitness_fn
        self.selection_fn = selection_fn
        self.mating_fn = mating_fn
        self.reproduction_fn = reproduction_fn
        self.mutation_fn = mutation_fn
        self.adaptive_population = adaptive_population
        self.elitism = elitism
        self.adaptive_mutation = adaptive_mutation

    def fit(self, X, y):
        gen = 1
        population = self.initialization_fn._populate(X)
        while True:
            # Evaluate their fitness
            population = self.fitness_fn._evaluate(population, X, y)

            # Check if the population is empty, is so, stop searching
            if isinstance(population, type(None)):
                print('Population Extincted')
                break

            # Preserves and drop elites before selection
            if self.elitism != None:
                population = self.elitism._preserve_elitism(population)
                population = self.elitism._drop_elite(population)

            # Select winners from the current generation
            if type(self.selection_fn) == BoltzmannSelection:
                population = self.selection_fn._select(population, self.fitness_fn.best_fitness_, gen, self.n_gen)

            else:
                population = self.selection_fn._select(population)

            # Merge elite back to the population for reproduction
            if self.elitism != None:
                population = self.elitism._preserve_elitism(population)

            # Adaptive population control
            if self.adaptive_population != None:
                if isinstance(self.adaptive_population.pop_cap, int):
                    pop_ratio = self.adaptive_population._control(population)
                    # update pop ratio of mating function
                    self.mating_fn.pop_ratio = pop_ratio

            # Check if the population is empty, is so, stop searching
            if isinstance(population, type(None)):
                print('Population Extincted')
                break

            population = self.mating_fn._pair(population)

            if population == None:
                cuerrent_time = datetime.now().strftime("%H:%M:%S")
                warnings.warn(f'[{cuerrent_time}] Early Stopping Warning: Early stopping triggered.')
                break

            population = self.reproduction_fn._reproduce(population, gen)

            # Drop elites before evaluation
            if self.elitism != None:
                population = self.elitism._drop_elite(population)

            # Mutation control
            if self.adaptive_mutation != None:
                self.mutation_fn.epsilon = self.adaptive_mutation._adaptive_mutation_rate(population)

            # Check if the population is empty, is so, stop searching
            if isinstance(population, type(None)):
                print('Population Extincted')
                break

            population = self.mutation_fn._mutate(population)

            # Check if the population is empty, is so, stop searching
            if isinstance(population, type(None)):
                print('Population Extincted')
                break

            gen += 1
            if gen == self.n_gen:
                break

        # Store attributes
        self.best_params_ = self.fitness_fn.best_params_
        self.best_fitness_ = self.fitness_fn.best_fitness_
        self.max_fitness_records_ = self.fitness_fn.max_fitness_records
        self.mean_fitness_records_ = self.fitness_fn.mean_fitness_records
        self.preselection_population_size_ = self.selection_fn.preselection_population_size
        self.survived_population_size = self.selection_fn.survived_population_size
