import numpy as np
import pandas as pd
from evolearn.hyperparameter_tuning.initialization import Genes
from evolearn.hyperparameter_tuning.evaluation import FitnessFunction
from evolearn.hyperparameter_tuning.selection import (RankSelection,
                                                      RouletteWheelSelection,
                                                      SteadyStateSelection,
                                                      TournamentSelection,
                                                      StochasticUniversalSampling,
                                                      BoltzmannSelection
                                                      )

from evolearn.hyperparameter_tuning.mating import MatingFunction
from evolearn.hyperparameter_tuning.reproduction import (KPointCrossover,
                                                         LinearCombinationCrossover,
                                                         FitnessProportionateAverage
                                                         )

from evolearn.hyperparameter_tuning.mutation import (Boundary,
                                                     Shrink
                                                     )

from evolearn.hyperparameter_tuning.environment import (AdaptiveReproduction,
                                                        AdaptiveMutation,
                                                        Elitism
                                                        )

from evolearn.hyperparameter_tuning.experimental.reproduction import AvergeReproduction

class GenesSearchCV:
    """Genetic hyperparameter_tuning over hyper parameters.
    GenesSearchCV implements a "fit" method.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    GenesSearchCV will reproduce new parameter values in each generation. The
    evolution processes is mostly probabilistic.

    Parameters
    ----------
    n_gen : int
       Maximum number of generation (or loop) GenesSearchCV will run.

    initialization_fn : hyperparameter_tuning.initialization.Genes class
       Class object to generate solution candidates.

    fitness_fn : hyperparameter_tuning.evaluation.FitnessFunction
       Class object to evalute the fitness of solution candidates.

    selection_fn : can either be
       - hyperparameter_tuning.selection.RankSelection,
       - hyperparameter_tuning.selection.RouletteWheelSelection,
       - hyperparameter_tuning.selection.SteadyStateSelection,
       - hyperparameter_tuning.selection.TournamentSelection,
       - hyperparameter_tuning.selection.StochasticUniversalSampling,
       - hyperparameter_tuning.selection.BoltzmannSelection

       Class object to evalute the fitness of solution candidates.

    mating_fn : hyperparameter_tuning.mating.MatingFunction
       Class object to pair the solution candidates for reproduction.

    reproduction_fn : can either be
       - hyperparameter_tuning.reproduction.KPointCrossover,
       - hyperparameter_tuning.reproduction.LinearCombinationCrossover,
       - hyperparameter_tuning.reproduction.FitnessProportionateAverage

       Class object to reproduce child population.

    mutation_fn : can either be
       - hyperparameter_tuning.mutation.Boundary,
       - hyperparameter_tuning.mutation.Shrink

       Class object to mutate the child population.

    adaptive_population : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mating rate of the mating_fn.

    elitism : hyperparameter_tuning.environment.Elitism, default: None
       Class object to perform elites selection, ace comparison and elites' traits induction.

    adaptive_mutation : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mutation probaility of the mutation_fn.

    Attributes
    ----------
    n_gen : int
       Maximum number of generation (or loop) GenesSearchCV will run.

    initialization_fn : hyperparameter_tuning.initialization.Genes class
       Class object to generate solution candidates.

    fitness_fn : hyperparameter_tuning.evaluation.FitnessFunction
       Class object to evalute the fitness of solution candidates.

    selection_fn : can either be
       - hyperparameter_tuning.selection.RankSelection,
       - hyperparameter_tuning.selection.RouletteWheelSelection,
       - hyperparameter_tuning.selection.SteadyStateSelection,
       - hyperparameter_tuning.selection.TournamentSelection,
       - hyperparameter_tuning.selection.StochasticUniversalSampling,
       - hyperparameter_tuning.selection.BoltzmannSelection

       Class object to evalute the fitness of solution candidates.

    mating_fn : hyperparameter_tuning.mating.MatingFunction
       Class object to pair the solution candidates for reproduction.

    reproduction_fn : can either be
       - hyperparameter_tuning.reproduction.KPointCrossover,
       - hyperparameter_tuning.reproduction.LinearCombinationCrossover,
       - hyperparameter_tuning.reproduction.FitnessProportionateAverage

       Class object to reproduce child population.

    mutation_fn : can either be
       - hyperparameter_tuning.mutation.Boundary,
       - hyperparameter_tuning.mutation.Shrink

       Class object to mutate the child population.

    adaptive_population : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mating rate of the mating_fn.

    elitism : hyperparameter_tuning.environment.Elitism, default: None
       Class object to perform elites selection, ace comparison and elites' traits induction.

    adaptive_mutation : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
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

    def __init__(self, n_gen: int, initialization_fn, fitness_fn, selection_fn, mating_fn, reproduction_fn, mutation_fn, adaptive_population=None, elitism=None, adaptive_mutation=None):
        if type(initialization_fn) not in [Genes]:
            raise Exception('Incorrect arguemnt for initialization_fn.')

        if type(fitness_fn) not in  [FitnessFunction]:
            raise Exception('Incorrect arguemnt for fitness_fn.')

        if type(selection_fn) not in [RankSelection, RouletteWheelSelection, SteadyStateSelection, TournamentSelection, StochasticUniversalSampling, BoltzmannSelection]:
            raise Exception('Incorrect arguemnt for selection_fn.')

        if type(mating_fn) not in [MatingFunction]:
            raise Exception('Incorrect arguemnt for mating_fn.')

        if type(reproduction_fn) not in [KPointCrossover, LinearCombinationCrossover, AvergeReproduction, FitnessProportionateAverage]:
            raise Exception('Incorrect arguemnt for reproduction_fn.')

        if type(mutation_fn) not in [Boundary, Shrink]:
            raise Exception('Incorrect arguemnt for mutation_fn.')

        if type(adaptive_population) not in [AdaptiveReproduction, type(None)]:
            raise Exception('Incorrect arguemnt for adaptive_population.')

        if type(elitism) not in [Elitism, type(None)]:
            raise Exception('Incorrect arguemnt for elitism.')

        if type(adaptive_mutation) not in [AdaptiveMutation, type(None)]:
            raise Exception('Incorrect arguemnt for adaptive_mutation.')

        if (type(reproduction_fn) in [LinearCombinationCrossover, AvergeReproduction, FitnessProportionateAverage]) and (initialization_fn.all_str):
            raise Exception('LinearCombinationCrossover and FitnessProportionateAverage only works if all hyperparameters in given searching space are numeric.')

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
        """Run fit on the estimator with randomly drawn parameters.
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).
        """

        gen = 1

        # Initialize the population
        population = self.initialization_fn._populate()
        while True:
            # Check if the population is empty, is so, stop searching
            if isinstance(population, type(None)):
                print('Population Extincted')
                break

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

            # Forming pairs of parents
            population = self.mating_fn._pair(population)

            # Reproduce next generation
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

            # Mutate the next generation
            population = self.mutation_fn._mutate(population)

            # If maximum number of generation reached, stop searching
            if gen >= self.n_gen:
                break

            gen += 1

        # Store attributes
        self.best_params_ = self.fitness_fn.best_params_
        self.best_fitness_ = self.fitness_fn.best_fitness_
        self.max_fitness_records_ = self.fitness_fn.max_fitness_records
        self.mean_fitness_records_ = self.fitness_fn.mean_fitness_records
        self.preselection_population_size_ = self.selection_fn.preselection_population_size
        self.survived_population_size = self.selection_fn.survived_population_size


class NSGAII:
    """Genetic hyperparameter_tuning over hyper parameters.
    GenesSearchCV implements a "fit" method.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    GenesSearchCV will reproduce new parameter values in each generation. The
    evolution processes is mostly probabilistic.

    Parameters
    ----------
    n_gen : int
       Maximum number of generation (or loop) GenesSearchCV will run.

    initialization_fn : hyperparameter_tuning.initialization.Genes class
       Class object to generate solution candidates.

    fitness_fn : hyperparameter_tuning.evaluation.FitnessFunction
       Class object to evalute the fitness of solution candidates.

    selection_fn : can either be
       - hyperparameter_tuning.selection.RankSelection,
       - hyperparameter_tuning.selection.RouletteWheelSelection,
       - hyperparameter_tuning.selection.SteadyStateSelection,
       - hyperparameter_tuning.selection.TournamentSelection,
       - hyperparameter_tuning.selection.StochasticUniversalSampling,
       - hyperparameter_tuning.selection.BoltzmannSelection

       Class object to evalute the fitness of solution candidates.

    mating_fn : hyperparameter_tuning.mating.MatingFunction
       Class object to pair the solution candidates for reproduction.

    reproduction_fn : can either be
       - hyperparameter_tuning.reproduction.KPointCrossover,
       - hyperparameter_tuning.reproduction.LinearCombinationCrossover,
       - hyperparameter_tuning.reproduction.FitnessProportionateAverage

       Class object to reproduce child population.

    mutation_fn : can either be
       - hyperparameter_tuning.mutation.Boundary,
       - hyperparameter_tuning.mutation.Shrink

       Class object to mutate the child population.

    adaptive_population : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mating rate of the mating_fn.

    elitism : hyperparameter_tuning.environment.Elitism, default: None
       Class object to perform elites selection, ace comparison and elites' traits induction.

    adaptive_mutation : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mutation probaility of the mutation_fn.

    Attributes
    ----------
    n_gen : int
       Maximum number of generation (or loop) GenesSearchCV will run.

    initialization_fn : hyperparameter_tuning.initialization.Genes class
       Class object to generate solution candidates.

    fitness_fn : hyperparameter_tuning.evaluation.FitnessFunction
       Class object to evalute the fitness of solution candidates.

    selection_fn : can either be
       - hyperparameter_tuning.selection.RankSelection,
       - hyperparameter_tuning.selection.RouletteWheelSelection,
       - hyperparameter_tuning.selection.SteadyStateSelection,
       - hyperparameter_tuning.selection.TournamentSelection,
       - hyperparameter_tuning.selection.StochasticUniversalSampling,
       - hyperparameter_tuning.selection.BoltzmannSelection

       Class object to evalute the fitness of solution candidates.

    mating_fn : hyperparameter_tuning.mating.MatingFunction
       Class object to pair the solution candidates for reproduction.

    reproduction_fn : can either be
       - hyperparameter_tuning.reproduction.KPointCrossover,
       - hyperparameter_tuning.reproduction.LinearCombinationCrossover,
       - hyperparameter_tuning.reproduction.FitnessProportionateAverage

       Class object to reproduce child population.

    mutation_fn : can either be
       - hyperparameter_tuning.mutation.Boundary,
       - hyperparameter_tuning.mutation.Shrink

       Class object to mutate the child population.

    adaptive_population : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
       Class object to adaptively change the mating rate of the mating_fn.

    elitism : hyperparameter_tuning.environment.Elitism, default: None
       Class object to perform elites selection, ace comparison and elites' traits induction.

    adaptive_mutation : hyperparameter_tuning.environment.AdaptiveReproduction, default: None
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

    def __init__(self, n_gen: int, initialization_fn, fitness_fn, selection_fn, mating_fn, reproduction_fn, mutation_fn, adaptive_population=None, elitism=None, adaptive_mutation=None):
        if type(initialization_fn) not in [Genes]:
            raise Exception('Incorrect arguemnt for initialization_fn.')

        if type(fitness_fn) not in  [FitnessFunction]:
            raise Exception('Incorrect arguemnt for fitness_fn.')

        if type(selection_fn) not in [RankSelection, RouletteWheelSelection, SteadyStateSelection, TournamentSelection, StochasticUniversalSampling, BoltzmannSelection]:
            raise Exception('Incorrect arguemnt for selection_fn.')

        if type(mating_fn) not in [MatingFunction]:
            raise Exception('Incorrect arguemnt for mating_fn.')

        if type(reproduction_fn) not in [KPointCrossover, LinearCombinationCrossover, AvergeReproduction, FitnessProportionateAverage]:
            raise Exception('Incorrect arguemnt for reproduction_fn.')

        if type(mutation_fn) not in [Boundary, Shrink]:
            raise Exception('Incorrect arguemnt for mutation_fn.')

        if type(adaptive_population) not in [AdaptiveReproduction, type(None)]:
            raise Exception('Incorrect arguemnt for adaptive_population.')

        if type(elitism) not in [Elitism, type(None)]:
            raise Exception('Incorrect arguemnt for elitism.')

        if type(adaptive_mutation) not in [AdaptiveMutation, type(None)]:
            raise Exception('Incorrect arguemnt for adaptive_mutation.')

        if (type(reproduction_fn) in [LinearCombinationCrossover, AvergeReproduction, FitnessProportionateAverage]) and (initialization_fn.all_str):
            raise Exception('LinearCombinationCrossover and FitnessProportionateAverage only works if all hyperparameters in given searching space are numeric.')

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
        """Run fit on the estimator with randomly drawn parameters.
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).
        """

        gen = 1

        # Initialize the population
        population = self.initialization_fn._populate()
        while True:
            # Check if the population is empty, is so, stop searching
            if isinstance(population, type(None)):
                print('Population Extincted')
                break

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

            # Forming pairs of parents
            population = self.mating_fn._pair(population)

            # Reproduce next generation
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

            # Mutate the next generation
            population = self.mutation_fn._mutate(population)

            # If maximum number of generation reached, stop searching
            if gen >= self.n_gen:
                break

            gen += 1

        # Store attributes
        self.best_params_ = self.fitness_fn.best_params_
        self.best_fitness_ = self.fitness_fn.best_fitness_
        self.max_fitness_records_ = self.fitness_fn.max_fitness_records
        self.mean_fitness_records_ = self.fitness_fn.mean_fitness_records
        self.preselection_population_size_ = self.selection_fn.preselection_population_size
        self.survived_population_size = self.selection_fn.survived_population_size