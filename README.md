![Logo](https://github.com/HindyDS/evo-learn/blob/main/logo/evolearn.png)

Evolutionary Algorithm For Machine Learning

Installation
------------

To use evolearn, first install it using pip:

    pip install evolearn

Genetic Optimization CV
----------------

To perform hyperparameter tuning using genetic algoritm,
you need to first import other modules from 

1) ``evolearn.hyperparameter_tuning.initialization``
2) ``evolearn.hyperparameter_tuning.evaluation``
3) ``evolearn.hyperparameter_tuning.selection``
4) ``evolearn.hyperparameter_tuning.mating``
5) ``evolearn.hyperparameter_tuning.reproduction``
6) ``evolearn.hyperparameter_tuning.mutation``
7) ``evolearn.hyperparameter_tuning.environment`` (optional)
8) ``evolearn.hyperparameter_tuning.genetic_hyperparameter_tuning`` 

Although the modules from ``environment`` are optional for you to determine to
use them in your search or not, the searching might end up stopping early or not 
finding the ideal results. These modules can help to prevent pre-mature convergence
and also control other hyperparameters for GA.

For example:

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
    from evolearn.hyperparameter_tuning.genetic_hyperparameter_tuning import GenesSearchCV
    from sklearn.ensemble import RandomForestRegressor
    search_space_rf = {
              'max_depth':(1, 16, 'uniform'),
              'n_estimators':(100, 1000, 'uniform'),
              'criterion':('squared_error', 'absolute_error', 'poisson')
          }  
    opt = GenesSearchCV(
          n_gen=10,
          initialization_fn=Genes(search_space=search_space_rf, pop_size=30),
          fitness_fn=FitnessFunction(
              estimator=RandomForestRegressor(n_jobs=-1),
              cv=3,
              scoring='neg_mean_absolute_error',
          ),
          selection_fn=StochasticUniversalSampling(.7),
          mating_fn=MatingFunction(increst_prevention=False),
          reproduction_fn=KPointCrossover(1),
          mutation_fn=Shrink(),
          adaptive_population=AdaptiveReproduction(10),
          elitism=Elitism(),
          adaptive_mutation=AdaptiveMutation()
      )   
    opt.fit(X_train, y_train)
  
    Max Fitness: -2023.200579609583
    {'max_depth': 5, 'n_estimators': 561, 'criterion': 'absolute_error'}


The choices of ``selection_fn``, ``reproduction_fn``, ``mutation_fn`` are
actually up to your personal preference. One can pick what they believe
are most benefit to their searching preocess.


Genetic Feature Selection
-------------------------

To perform feature selection using genetic algoritm,
you need to first import other modules from 

1) ``evolearn.feature_selection.initialization``
2) ``evolearn.feature_selection.evaluation``
3) ``evolearn.feature_selection.selection``
4) ``evolearn.feature_selection.mating``
5) ``evolearn.feature_selection.reproduction``
6) ``evolearn.feature_selection.mutation``
7) ``evolearn.feature_selection.environment`` (optional)
8) ``evolearn.feature_selection.genetic_feature_selection`` 

The modules looks similar to those modules from the 
``GenesSearchCV`` section, but in fact their internal mechanisim 
work slightly differently. You need to be ware of importing the 
wrong modules when using genetic feature selection.

For example:

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
    from evolearn.feature_selection.reproduction import KPointCrossover
    from evolearn.feature_selection.mutation import (BitStringMutation,
                                                    ExchangeMutation,
                                                    ShiftMutation
                                                    )

    from evolearn.feature_selection.environment import (AdaptiveReproduction,
                                                    AdaptiveMutation,
                                                    Elitism
                                                    )

    from evolearn.feature_selection.genetic_feature_selection import GeneticFeatureSelectionCV
    from sklearn.ensemble import RandomForestRegressor
    opt = GeneticFeatureSelectionCV(
       n_gen=10,
       initialization_fn=Genes(pop_size=50),
       fitness_fn=FitnessFunction(
           estimator=RandomForestRegressor(n_jobs=-1),
           cv=3,
           scoring='neg_mean_absolute_error'
       ),
       selection_fn=RouletteWheelSelection(.7),
       mating_fn=MatingFunction(),
       reproduction_fn=KPointCrossover(k=4),
       mutation_fn=BitStringMutation(),
       adaptive_population=None,
       elitism=None,
       adaptive_mutation=None
       )

    opt.fit(X_train, y_train)
    print(opt.best_fitness_)
    print(opt.best_params_)

    -2797.7245589631652
    {'age': True, 'sex': False, 'bmi': True, 'children': True, 'smoker': True, 'region': False}