import warnings
from datetime import datetime

class MatingFunction:
    '''MatingFunction is function that use to matching solution candidates
        in pairs for experimental.

        Parameters
        ----------
        pop_ratio : float
        Fraction that determine the number of couples produced by the MatingFunction.

        increst_prevention : boolean
        if True, candidates sharing the same parents will be paired together.
    '''

    def __init__(self, pop_ratio=1, increst_prevention=True):
        if 0 > pop_ratio:
            raise Exception('MutaingFunction argument "pop_ratio" must positive float.')

        if not isinstance(increst_prevention, bool):
            raise Exception('MatingFunction argument "increst_prevention" must be boolean.')

        if pop_ratio < 2:
            cuerrent_time = datetime.now().strftime("%H:%M:%S")
            warnings.warn(
                f'[{cuerrent_time}] Population Decline Warning: "pop_ratio" of less than 2 might be too small for maintaining population size. This might lead to not having enough couples for experimental. Please consider a larger Initial population size and increase the number of survivors.')

        if increst_prevention:
            cuerrent_time = datetime.now().strftime("%H:%M:%S")
            warnings.warn(f'[{cuerrent_time}] Population Decline Warning: "increst_prevention" is set to "True". This might lead to not having enough couples for experimental. Please consider a larger Initial population size and increase the number of survivors.')

        self.pop_ratio = pop_ratio
        self.increst_prevention = increst_prevention

    def _dominant_pairing(self, population):
        '''Parameters
        ----------
        population : {array-like, sparse matrix} of shape (n_samples, n_features)
        Fitness of each candidate.
        '''

        pop_ratio = self.pop_ratio
        increst_prevention = self.increst_prevention

        pop_ratio = int(pop_ratio * len(population))

        pairs = []
        # Select the most fitness dominant individual
        population.sort_values('fitness', ascending=False, inplace=True)

        # pairing with the rest of the candidates
        pair_count = 0
        for p1 in range(len(population)):
            for p2 in range(len(population)):
                if p1 != p2:
                    chromosome1 = population.iloc[[p1]]
                    chromosome2 = population.iloc[[p2]]

                    # If increst then skip
                    if increst_prevention and (chromosome1['parents'].values[0] != 'None' and chromosome2['parents'].values[0] != 'None'):
                        if chromosome1['parents'].values[0] == chromosome2['parents'].values[0]:
                            continue

                    chromosome1 = chromosome1.drop('fitness', axis=1).copy()
                    chromosome2 = chromosome2.drop('fitness', axis=1).copy()

                    if 'parents' in chromosome1.columns:
                        chromosome1.drop('parents', axis=1, inplace=True)

                    if 'parents' in chromosome2.columns:
                        chromosome2.drop('parents', axis=1, inplace=True)

                    pairs.append((chromosome1, chromosome2))
                    pair_count += 1

                if pair_count >= pop_ratio:
                    break

        return pairs
