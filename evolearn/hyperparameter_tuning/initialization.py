import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Optional, Union

class Genes:
    """Internal function that generate a pool of genes (search space) according to
       the given parameters.

       Parameters
       ----------
       search_space : dict
       Where keys are parameter names (strings)
       and values are int, float or str.
       Represents search space
       over parameters of the provided estimator.

       pop_size : int
       Size of the initial population.

       """

    def __init__(self, search_space: dict, pop_size: int):
        cuerrent_time: str
        if pop_size < 10:
            cuerrent_time = datetime.now().strftime("%H:%M:%S")
            warnings.warn(f'[{cuerrent_time}] Low Population Warning: Small initial population size maybe cause premature convergence. Please consider initializing a larger population size.')

        def _is_all_str(dictionary):
            for v in dictionary.values():
                if all(isinstance(item, str) for item in v):
                    return True

            return False

        self.all_str = _is_all_str(search_space)
        self.search_space = search_space
        self.pop_size = pop_size


    def _make_genes_numeric(self, lower: Optional[Union[int, float, str]], upper: Optional[Union[int, float, str]], prior: str, pop_size: int) -> pd.DataFrame:
        """Generate numerical values for solution candidates according to given statistical distribution.

        Parameters
        ----------
        lower : integer
        Defines the lower boundaries of the distributions.

        upper : integer
        Defines the upper boundaries of the distributions.

        pop_size : integer
        A parameter to control the size of the population.

        prior : string
        A parameter that defines the sampling distribution.
        """

        allele: np.array
        mu: float
        sigma: float
        genes: Union[float, list]

        # Check if upper is bigger than lower
        if lower >= upper:
            raise Exception('Upper must be larger than lower boundary.')

        # Check if sampling distribution available
        if prior not in ['normal', 'uniform', 'log-normal']:
            raise Exception('Prior for search space must be "normal", "uniform", "log-normal".')

        # Obtains mu and sigma from the interval
        allele = np.arange(lower, upper)
        mu = allele.mean()
        sigma = allele.std()

        # Sampling from normal distribution
        if prior == 'normal':
            genes = np.random.normal(loc=mu, scale=sigma, size=pop_size)

        # Sampling from uniform distribution
        elif prior == 'uniform':
            genes = np.random.uniform(low=lower, high=upper, size=pop_size)

        # Sampling from log-normal distribution
        elif prior == 'log-normal':
            genes = np.random.lognormal(mu=mu, sigma=sigma, size=pop_size)

        genes = (genes - genes.min()) / (genes.max() - genes.min()) * (upper - lower) + lower

        if isinstance(lower, int) and isinstance(upper, int):
            genes = [int(round(i)) for i in genes]

        return pd.DataFrame(genes, dtype='O')

    def _make_genes_categorical(self, categories: tuple, pop_size: int) -> pd.DataFrame:
        """Randomly generate categorical values for solution candidates.

        Parameters
        ----------
        categories : tuple
        Contains all possible categories for particular genes.

        pop_size : int
        A parameter to control the size of the population.
        """
        categorical_genes: Union[np.array, pd.DataFrame]

        categorical_genes = np.random.choice(a=categories, size=pop_size)
        categorical_genes = pd.DataFrame(categorical_genes)

        return categorical_genes


    def _populate(self) -> pd.DataFrame:
        """Generate initial population."""

        categorical_genes_columns: list
        categorical_genes: Union[list, pd.DataFrame]
        numerical_genes: Union[list, pd.DataFrame]
        numerical_gene: pd.DataFrame
        is_lower_numeric: bool
        is_upper_numeric: bool

        # Check if all are strings, if so, then categorical genes
        categorical_genes_columns = []
        categorical_genes = []
        for hyperparameter, hyperparameter_range in self.search_space.items():
            if all([isinstance(i, str) for i in hyperparameter_range]):
                categorical_genes_columns.append(hyperparameter)
                categorical_genes.append(self._make_genes_categorical(categories=hyperparameter_range, pop_size=self.pop_size))

        if not categorical_genes == []:
            categorical_genes = pd.concat(categorical_genes)
            categorical_genes.columns = [c for c in self.search_space.keys() if c in categorical_genes_columns]


        # If lower and upper are numeric, Make genes and store them in list as pandas dataframes
        numerical_genes = []
        for hyperparameter_range in self.search_space.values():
            lower = hyperparameter_range[0]
            upper = hyperparameter_range[1]
            prior = hyperparameter_range[2]

            # Two sets of conditions to check if both lower and upper are numeric
            is_lower_numeric = (isinstance(lower, int) or isinstance(lower, float))
            is_upper_numeric = (isinstance(upper, int) or isinstance(upper, float))

            if is_lower_numeric and is_upper_numeric:
                numerical_gene = self._make_genes_numeric(lower=lower, upper=upper, prior=prior, pop_size=self.pop_size)
                numerical_genes.append(numerical_gene)

        if not numerical_genes == []:
            # Concatenate the chromosome into population
            numerical_genes = pd.concat(numerical_genes, axis=1)
            numerical_genes_columns = [i for i in self.search_space.keys() if i not in categorical_genes_columns]
            numerical_genes.columns = numerical_genes_columns

        # If both not empty, then return combined pandas dataframe
        if isinstance(numerical_genes, pd.DataFrame) and isinstance(categorical_genes, pd.DataFrame):
            return pd.concat([numerical_genes, categorical_genes], axis=1)

        # If categorical_genes is empty, then return only the numerical_genes
        elif isinstance(numerical_genes, pd.DataFrame) and isinstance(categorical_genes, list):
            return numerical_genes

        # If numerical_genes is empty, then return only the categorical_genes
        elif isinstance(numerical_genes, list) and isinstance(categorical_genes, pd.DataFrame):
            return categorical_genes

