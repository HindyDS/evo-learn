import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

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
        self.memo = {}
        self.best_fitness_ = None
        self.max_fitness_records = []
        self.mean_fitness_records = []

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

        phenotype: pd.Series
        fitness: float

        # Store parents, remove parents and fitness
        if '__parents__' in population.columns:
            parents_records = population['__parents__']
            population.drop('__parents__', axis=1, inplace=True)

        if '__fitness__' in population.columns:
            population.drop('__fitness__', axis=1, inplace=True)

        def _sub_evaluate(p):
            fitness_records = []
            phenotype = pd.Series(X.columns)[p]
            if str(phenotype.to_list()) not in self.memo:
                fitness = cross_val_score(estimator=self.estimator, cv=self.cv, X=X[phenotype], y=y, scoring=self.scoring).mean()
                fitness_records.append(fitness)
                self.memo[str(phenotype.to_list())] = fitness

            else:
                fitness_records.append(self.memo[str(phenotype.to_list())])

            return fitness_records[0], self.memo

        fitness_records_and_memo = Parallel(n_jobs=-1)(delayed(_sub_evaluate)(p) for p in population.values)
        fitness_records = [i[0] for i in fitness_records_and_memo]
        memo = [i[1] for i in fitness_records_and_memo]

        result = {}
        for d in memo:
            result.update(d)

        self.memo = result

        population['__parents__'] = parents_records
        population['__fitness__'] = fitness_records
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