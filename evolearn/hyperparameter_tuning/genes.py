import numpy as np
from typing import Optional

def _make_genes(lower: Optional[int, float, str], upper: Optional[int, float, str], pop_size: int, prior: str) -> list:
    """Internal function that generate a pool of genes (search space) according to
       the given parameters.

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
    mu: np.float64
    sigma: np.float64
    prior: str
    genes: Union[float, list]

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
        genes = np.random.lognormal(mean=mu, sigma=sigma)

    genes = (genes - genes.min()) / (genes.max() - genes.min()) * (upper - lower) + lower

    if isinstance(lower, int) and isinstance(upper, int):
        genes = [int(round(i)) for i in genes]

    return genes
