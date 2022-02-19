from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Optional, Union

class FitnessFunction:
    """Evaluate the fitness of the population.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each search point.
        This object is assumed to implement the scikit-learn estimator api.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

     cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

     scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    """
    def __init__(self, estimator: BaseEstimator, cv: int, scoring: str):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.best_fitness_ = None
        self.best_params_ = None
        self.mean_fitness_records = []
        self.max_fitness_records = []

    def _evaluate(self, population: pd.DataFrame, X, y) -> pd.DataFrame:
        """Internal function that evaluate the fitness of each in individual candidate.

           Parameters
           ----------

           population : {array-like, sparse matrix} of shape (n_samples, n_features)
               Solution candidates in a list of tuples

           estimator : estimator object implementing ‘fit’
               The object to use to fit the data.

           scoring : string, callable or None, default=None
               A string (see model evaluation documentation) or a scorer callable
               object / function with signature scorer(estimator, X, y).
               If None, the score method of the estimator is used.

            X : {array-like, sparse matrix} of shape (n_samples, n_features)
               Training data.

            y : array-like of shape (n_samples,) or (n_samples, n_targets)
               Target values. Will be cast to X’s dtype if necessary.
           """

        population: Union[list, pd.DataFrame]

        # Store the datatype of the genes
        genes_dtype = population.dtypes.values

        # Turn chromosomes to list of tuples
        population = [{k: ([int(round(v))] if t in ['int64', 'int32'] else [v]) for (k, v, t) in
                      zip(population.columns, population.iloc[i], genes_dtype)} for i in range(len(population))]

        def _get_fitness(c):
            if '__parents__' not in c:
                p = 'None'

            else:
                p = c['__parents__']
                c.pop('__parents__')

            if '__fitness__' in c:
                c.pop('__fitness__')

            try:
                gs = GridSearchCV(estimator=self.estimator, param_grid=c, cv=self.cv, scoring=self.scoring)
                gs.fit(X, y)
                c['__fitness__'] = gs.best_score_
                c['__parents__'] = p

                return pd.DataFrame(c)

            except:
                pass

        # Try out each solution and store their fitness
        try:
            population = pd.concat(Parallel(n_jobs=-1)(delayed(_get_fitness)(p) for p in population))

        except ValueError:
            return None

        population.sort_values('__fitness__', ascending=False, inplace=True)
        population.dropna(inplace=True)
        population.index = range(len(population))

        # If better candidate found, update best fitness and best parameters
        if (self.best_fitness_ == None) or (population['__fitness__'].max() > self.best_fitness_):
            best_params = population.sort_values('__fitness__', ascending=False).iloc[0]
            best_params = best_params.drop(['__parents__', '__fitness__'])
            self.best_params_ = {k:v for k, v in zip(best_params.index, best_params.values)}
            self.best_fitness_ = population['__fitness__'].max()

        mean_fitness = population['__fitness__'].mean()
        self.max_fitness_records.append(self.best_fitness_)
        self.mean_fitness_records.append(mean_fitness)

        return population
