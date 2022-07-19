import numpy as np

class FitnessEvaluator:
    """
    The Fitness Evaluator is to check the fitness of solutions against the reference MFA from the Coordinator,
    as well as selecting the individuals for the next generation.
    """

    def extractFitnessValues(self, result):
        # Get the emission value for fitness comparison
        emissionDict = result[8]
        emissionValuesDomestic = emissionDict["total_emissions_domestic"]
        emissionValue = emissionValuesDomestic[-1]

        # Get the energy value for fitness comparison
        energyDict = result[7]
        energyProductionTimeSeries = energyDict["total_energy"]
        energyValue = energyProductionTimeSeries[-1]

        return emissionValue, energyValue

    def calculateFitnessFinalYear(self, solution_result, baseline_result):
        baselineEmission, baselineEnergy = self.extractFitnessValues(baseline_result)
        solutionEmission, solutionEnergy = self.extractFitnessValues(solution_result)

        if solutionEnergy < 0 or solutionEmission < 0:
            return 0

        changeEmission = baselineEmission / solutionEmission
        changeEnergy = baselineEnergy / solutionEnergy

        res = changeEmission * changeEnergy
        return res

    def calculateFitnessMeanAllYears(self, solution_result, baseline_result):
        timeSeriesFitness = self.calculateFitnessTimeSeries(solution_result, baseline_result)
        mean = np.mean(timeSeriesFitness)
        return mean

    def extractFitnessValuesTimeSeries(self, data):
        emissionDict = data[8]
        emissionValuesDomestic = emissionDict["total_emissions_domestic"]

        energyDict = data[7]
        energyProductionTimeSeries = energyDict["total_energy"]

        return emissionValuesDomestic, energyProductionTimeSeries

    def calculateFitnessTimeSeries(self, solution, result_baseline):
        baselineEmissionTimeSeries, baselineEnergyTimeSeries = self.extractFitnessValuesTimeSeries(result_baseline)
        solutionEmissionTimeSeries, solutionEnergyTimeSeries = self.extractFitnessValuesTimeSeries(solution)

        changeEmissionTimeSeries = baselineEmissionTimeSeries / solutionEmissionTimeSeries
        changeEnergyTimeSeries = baselineEnergyTimeSeries / solutionEnergyTimeSeries
        resTimeSeries = changeEmissionTimeSeries * changeEnergyTimeSeries
        return resTimeSeries

    def sortByFitnessAsc(self, population):
        # an implementation of Insertion-sort, sort by fitness value descending

        l = population
        for i in range(1, len(l)):
            element = l[i]  # we store the element that we are currently looking at
            j = i - 1  # and we only start one element below that one

            fitnessValueSource = element.fitnessValue
            fitnessValueTarget = l[j].fitnessValue

            while fitnessValueSource < fitnessValueTarget and j >= 0:  # then we go towards the "left" of the list,
                # as long as our stored element is smaller, but only until we reach the first position

                l[j + 1] = l[j]  # swap the next smaller element one slot upwards
                l[j] = element  # and fill that one's position with our current element
                j -= 1  # then decrease the position

                fitnessValueTarget = l[j].fitnessValue

        return l

    # def linearRankSelectionNaive(self, generation): #these method did not work
    #     generation = list(range(0, 100))
    #     n = len(generation)
    #
    #     v = 1 / (n - 2.001)
    #
    #     selection = []
    #     for sourceIndividual in generation:
    #         a = np.random.uniform(0, v)
    #
    #         for targetIndividual in generation:
    #
    #                 rank = generation.index(targetIndividual) + 1  # start at 1 for the formula
    #                 probability_j = rank / (n * (n - 1))
    #
    #                 if probability_j <= a and targetIndividual not in selection:
    #                     selection.append(targetIndividual)
    #                     break
    #
    #     pass

    # def exponentialRankSelectionNaive(self, generation): #this one did not work either
    #     generation = list(range(0, 100))
    #     n = len(generation)
    #
    #     selection = []
    #
    #     c= (n * 2 * (n-1)) / (6*(n-1)+n)
    #
    #     for sourceIndividual in generation:
    #         a = np.random.uniform(1/9 * c, 2/c)
    #         for targetIndividual in generation:
    #
    #             rank = generation.index(targetIndividual) + 1  # start at 1 for the formula
    #             exponent = (-rank/c)
    #             probability_j = 1.0 * np.exp(exponent)
    #
    #             if probability_j <= a :
    #                 selection.append(targetIndividual)
    #                 break
    #
    #     pass

    def linearRankSelection(self, generation):
        """
        We need to find the best solutions of the generation to perform cross-over and mutation on them, so we select
        them. We will use a Linear Rank Selection (LRS), as:
        "it tries to overcome the drawback of premature convergence of the GA to a local optimum. It is
        based on the rank of individuals rather than on their fitness. The rank n is accorded to the best individual
        whilst the worst individual gets the rank 1." (Jebari, Khalid and Madiafi, Mohammed. “Selection methods for
        genetic algorithms”. In: International Journal of Emerging Sciences 3.4 (2013), pp. 333–344)
        """

        # Linear Rank Selection, beta and alpha mean how many of the most fit, and how many of the least fit individuals we desire
        betaRank = 2
        alphaRank = 2 - betaRank
        mu = len(generation)  # mu refers to the generation size

        # first we create a dictionary assigning to each individual a probability of being selected later on
        probabilityDict = {}

        for i in range(mu):  # rank of least fit =0, most fit = mu-1, fits with the for loop
            probability_i = alphaRank + (i / (mu - 1)) * (betaRank - alphaRank)
            probability_i = probability_i / mu

            individual = generation[i]
            probabilityDict[individual] = probability_i
        # the sum of all probabilities equals 1, this has been verified

        # now we sample a random uniform number between 0 and 1, and iterate over the probability dictionary,
        # as long as the cumulative Probability does not exceed the uniform sample, if we have reached that point,
        # we take that individual into the selection, and repeat as much as desired.
        selection = []
        for i in range(len(generation)):
            chosen = np.random.uniform(0,
                                       1)  # inspired by https://stackoverflow.com/questions/4113307/pythonic-way-to
            # -select-list-elements-with-different-probability
            cumulative = 0
            for individual in probabilityDict:
                probability = probabilityDict[individual]
                cumulative += probability
                if cumulative > chosen:
                    selection.append(individual)
                    break

        return selection
