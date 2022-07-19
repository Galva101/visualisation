import copy

from mfa.tests import baseline_pathways_test, mass_balance_test
from mfa.baseline_data_model import energy_intensity_calculation
from mfa.mfa_model import ModelMFA
from sensitivity_analysis import case_5_transformation, case_5_assign, case_5_transformation_inflow, \
    case_5_re_introduction_inflow, case_6_transformation, case_4_transformation
import config as cfg
from ga.fitness_evaluator import FitnessEvaluator

import numpy as np


def checkValueBoundaries(min, max, v):
    # to make sure a value does not exceed its boundaries
    if v < min:
        v = min
    if v > max:
        v = max
    return v


class Solution:
    """
    A solution receives an MFA chromosome, either from the baseline in the Coordinator at the beginning,
    or from a parent later on.
    """
    data = ""
    model_mfa = ""
    result = ""

    populationNumber = ""
    solutionNumber = ""

    emissionValue = 0
    energyValue = 0
    fitnessValue = 0

    chromosome = ""

    def __init__(self, data, populationNumber, solutionNumber):
        self.data = data
        self.solutionNumber = solutionNumber
        self.populationNumber = populationNumber

    def runSolutionMFA(self):
        print("Starting G" + str(self.populationNumber) + "S" + str(self.solutionNumber))

        self.model_mfa = ModelMFA(self.data)  # store this separately for the Mass Balance Test
        self.result = self.model_mfa.compilation_model()
        f = FitnessEvaluator()
        self.emissionValue, self.energyValue = f.extractFitnessValues(self.result)

        return self

    def massBalanceTest(self):
        mfaReference = self.model_mfa
        mass_balance_test(mfaReference.dyn_mfa_system.FlowDict, mfaReference.dyn_mfa_system.StockDict, self.data)
        print("Mass Balance Test over for G" + str(self.populationNumber) + "S" + str(self.solutionNumber) + "\n")

    def crossover(self, parentA, parentB):
        # Using a K-Point crossover function: generate k random points along the chromosome, alternate between them
        # when creating the children chromosomes

        numPoints = 5  # here we set how many crossover points we want
        crossoverPoints = []
        while len(crossoverPoints) < numPoints:
            index = np.random.randint(0, len(parentA))
            if index not in crossoverPoints:
                crossoverPoints.append(index)

        childA = []
        childB = []

        flip = False  # reverse the crossover direction later on
        for i in range(-1, len(parentA)):
            if i >= 0:  # this is done so that the first position can also be flipped around, if we only start at 0,
                # the first iteration will never be swapped, and only then will it check if 0 is a crossover point,
                # and reverse flip for the position 1
                if flip:  # perform a crossover
                    childA.append(parentB[i])
                    childB.append(parentA[i])
                else:  # perform no crossover
                    childA.append(parentA[i])
                    childB.append(parentB[i])

            if i in crossoverPoints:
                flip = not flip

        return childA, childB

    def initialiseChromosome(self):
        # Initialise the chromosome with random values of the given distributions

        year_LowerBoundary = 2025
        year_UpperBoundary = 2100

        percentage_LowerBoundary = 0
        percentage_UpperBoundary = 1

        populationEffPercent = np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)
        populationYear = np.random.randint(year_LowerBoundary, year_UpperBoundary)

        renovationCycleEffPercent = np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)
        renovationCycleYear = np.random.randint(year_LowerBoundary, year_UpperBoundary)

        energyIntensityEffPercent = np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)
        energyIntensityYear = np.random.randint(year_LowerBoundary, year_UpperBoundary)

        typeSplitSplitShareEffPercent = np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)
        typeSplitSplitShareYear = np.random.randint(year_LowerBoundary, year_UpperBoundary)

        dwellingEffPercent = np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)
        dwellingYear = np.random.randint(year_LowerBoundary, year_UpperBoundary)

        alphaStart = [np.random.randint(year_LowerBoundary, year_UpperBoundary),
                      np.random.randint(year_LowerBoundary, year_UpperBoundary)]
        alphaDuration = [np.random.randint(0, year_UpperBoundary - alphaStart[0]), np.random.randint(0,
                                                                                                     year_UpperBoundary -
                                                                                                     alphaStart[
                                                                                                         1])]  # two durations within the year intervals
        alphaEnd = [np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary),
                    np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)]

        possibleCases = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "5A", "6A"]
        alphaCase = np.random.choice(possibleCases)

        alphaReintroductionStart = np.random.randint(year_LowerBoundary, year_UpperBoundary)
        alphaReintroductionDuration = np.random.randint(year_LowerBoundary, year_UpperBoundary)
        alphaReintroductionEndRate = np.random.uniform(0, 1)

        chr = [
            populationEffPercent, populationYear,
            renovationCycleEffPercent, renovationCycleYear,
            energyIntensityEffPercent, energyIntensityYear,
            typeSplitSplitShareEffPercent, typeSplitSplitShareYear,
            dwellingEffPercent, dwellingYear,
            alphaStart, alphaDuration, alphaEnd, alphaCase,
            alphaReintroductionStart, alphaReintroductionDuration, alphaReintroductionEndRate
        ]

        self.chromosome = chr

    def overwriteChromosome(self, chromosome):
        self.chromosome = chromosome

    def mutateChromosome(self):
        # Mutate the chromosome by randomly changing values

        mutationRate = 0.15
        year_LowerBoundary = 2025
        year_UpperBoundary = 2100
        percentage_LowerBoundary = 0
        percentage_UpperBoundary = 1

        for i in range(len(self.chromosome)):
            if np.random.uniform(0, 1) < mutationRate:  # draw a uniform number and
                # if it is below the mutationRate, we mutate that value of the chromosome
                if i <= 9:  # these are the effectPercent and year Values
                    if i % 2 == 0:  # if i is even, we mutate the effectPercent
                        self.chromosome[i] = np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)
                    else:
                        self.chromosome[i] = np.random.randint(year_LowerBoundary,
                                                               year_UpperBoundary)  # if i is odd, we mutate the year

                if i == 10:  # this is the alphaStart
                    self.chromosome[i] = [np.random.randint(year_LowerBoundary, year_UpperBoundary),
                                          np.random.randint(year_LowerBoundary, year_UpperBoundary)]
                if i == 11:  # this is the alphaDuration
                    self.chromosome[i] = [np.random.randint(0, year_UpperBoundary - self.chromosome[10][0]),
                                          np.random.randint(0, year_UpperBoundary - self.chromosome[10][1])]
                    # referencing the alpha start first and second values

                if i == 12:  # this is the alphaEnd
                    self.chromosome[i] = [np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary),
                                          np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)]
                if i == 13:  # this is the alphaCase
                    possibleCases = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "5A", "6A"]
                    self.chromosome[i] = np.random.choice(possibleCases)
                if i == 14:  # this is the alphaReintroductionStart
                    self.chromosome[i] = np.random.randint(year_LowerBoundary, year_UpperBoundary)
                if i == 15:  # this is the alphaReintroductionDuration
                    self.chromosome[i] = np.random.randint(year_LowerBoundary, year_UpperBoundary)
                if i == 16:  # this is the alphaReintroductionEndRate
                    self.chromosome[i] = np.random.uniform(percentage_LowerBoundary, percentage_UpperBoundary)

        pass

    def applyChromosome(self):
        chr = self.chromosome

        populationEffPercent = chr[0]
        populationYear = chr[1]
        renovationCycleEffPercent = chr[2]
        renovationCycleYear = chr[3]
        energyIntensityEffPercent = chr[4]
        energyIntensityYear = chr[5]
        typeSplitSplitShareEffPercent = chr[6]
        typeSplitSplitShareYear = chr[7]
        dwellingEffPercent = chr[8]
        dwellingYear = chr[9]
        alphaStart = chr[10]
        alphaDuration = chr[11]
        alphaEnd = chr[12]
        alphaCase = chr[13]
        alphaReintroductionStart = chr[14]
        alphaReintroductionDuration = chr[15]
        alphaReintroductionEndRate = chr[16]

        self.mutatePopulation(populationEffPercent, populationYear)
        self.mutateRenovationCycle(renovationCycleEffPercent, renovationCycleYear)
        self.mutateEnergyIntensity(energyIntensityEffPercent, energyIntensityYear)
        self.mutateDwelling(dwellingEffPercent, dwellingYear)
        self.mutateTypesplitAndSplitshares(typeSplitSplitShareEffPercent, typeSplitSplitShareYear)

        # self.mutateAlphaParameterised(alphaStart, alphaDuration, alphaEnd, alphaCase,
        #                               alphaReintroductionStart, alphaReintroductionDuration, alphaReintroductionEndRate)

        pass

    def mutateRenovationCycle(self, effect_percent, year):  # Sensitivity Analysis Case 6

        data = self.data  # get the reference (and restore it from the previous iteration)
        baseline = copy.deepcopy(data)

        for e in cfg.CASE6:
            param = e.split('.')
            attr = getattr(data, param[0])  # e.g. 'private_vehicles'
            d: np.ndarray = attr[param[1]]
            y_start = cfg.END_YEAR + 1 - len(d)

            # Apply transformation to input data
            case_6_transformation(getattr(data, param[0])[param[1]],  # we restore `input_data`,
                                  year - y_start, effect_percent)  # so `d` no longer points to the right place

            energy_intensity = energy_intensity_calculation(
                data.residential_buildings['energy_intensity_resbuild_raw'],
                getattr(data, param[0])[param[1]],
                data.residential_buildings['types'],
                data.residential_buildings['index_python'],
                data.residential_buildings['cohort_start'],
                data.residential_buildings['cohort_end'],
                data.residential_buildings['start_time'],
                data.residential_buildings['cohort_number'],
                data.residential_buildings['renovation_states'],
                year - 1600)
            energy_intensity_resbuild = energy_intensity.calculation_energy_intensity_SA()
            energy_intensity_resbuild = energy_intensity_resbuild[year - 1600:501, :, :]
            data.residential_buildings['energy_intensity'] = baseline.residential_buildings['energy_intensity']
            data.residential_buildings['energy_intensity'][year - 1600:501, :, :] = energy_intensity_resbuild

        pass

    def mutateEnergyIntensity(self, effect_percent, year):  # Sensitivity Analysis Case 4
        input_data = self.data
        data = copy.deepcopy(input_data)  # we will change this input data

        for e in cfg.CASE4:
            param = e.split('.')
            attr = getattr(input_data, param[0])  # e.g. 'private_vehicles'
            d: np.ndarray = attr[param[1]]
            y_start = cfg.END_YEAR + 1 - len(d)
            splits = [i for i in data.residential_buildings['cohort_number']]  # e.g. [0, 1, 2, 3, 4, 5...]
            cohorts_starts = [i for i in data.residential_buildings['cohort_start']]
            cohorts_ends = [i for i in data.residential_buildings['cohort_end']]
            types = [i for i in data.residential_buildings['types'] - 1]  # e.g. [0, 1]

            for type in types:
                for split in splits:
                    print(f'Going through building type {type} cohort {split} of {e}')
                    min_energy = data.residential_buildings['minimum_energy_intensity']
                    case_4_transformation(getattr(input_data, param[0])[param[1]],  # we restore `input_data`,
                                          year - y_start, effect_percent, cohorts_starts, cohorts_ends, split,
                                          min_energy, type)  # so `d` no longer points to the right place
        pass

    def mutateTypesplitAndSplitshares(self, effect_percent, year):  # Sensitivity Analysis Case 2
        categories = cfg.CASE2

        data = self.data  # get the reference for easier coding below

        for category in categories:
            splitName = category.split(".")
            attr = getattr(data, splitName[0])  # Gets the dictionary with name before the "."
            d: np.ndarray = attr[splitName[
                1]]  # Gets from that dictionary the entries after the ".", for the first it is the (501, 4) dimensional array

            splits = [i for i in range(0, attr[splitName[1]].shape[1])]  # list of splits, e.g. [0, 1, 2, 3]
            startIndex = year - (2100 + 1 - len(d))

            for split in splits:
                print(f'Going through split {split} of {category}')
                print(f'Case 2 transformation(__, {year}, {effect_percent}, column={split})')
                for y in range(startIndex, d.shape[0]):  # shape: (len(years), len(splits))
                    # logging.debug(f'applying change to params in index: {y})')
                    # logging.debug(f'\t Old values: {data[y][0]} {data[y][1]} {data[y][2]} {data[y][3]}')
                    old = d[y][split]
                    d[y][split] *= effect_percent
                    change = (old - d[y][split]) / (d.shape[1] - 1)
                    splits = [i for i in range(0, d.shape[1]) if i != split]
                    for s in splits:
                        d[y][s] += change
                    # logging.debug(f'\t Adding {change} to each of other parameters')
                    # logging.debug(f'\t New values: {data[y][0]} {data[y][1]}  {data[y][2]}  {data[y][3]} ')
        pass

    def mutatePopulation(self, effect_percent, year):
        max = 10546327 #from the documentation
        min = 2100251

        attr = getattr(self.data, "population_1799_2100")

        for i in range(len(attr)):
            if i + 2100 - len(attr) >= year:  # add the shift back in because the years list is only 302 elements long
                oldValue = attr[i]
                newValue = oldValue * effect_percent
                newValue = checkValueBoundaries(min, max, newValue)
                attr[i] = newValue

        print("Population Mutated")

    def mutateDwelling(self, effect_percent, year):  # Sensitivity Analysis case 3
        categories = cfg.CASE3

        for category in categories:
            data = self.data  # get the reference (and restore it from the previous iteration)

            param = category.split('.')
            attr = getattr(data, param[0])
            d: np.ndarray = attr[param[1]]

            y_start = 2100 + 1 - len(d)  # debug check: should be 1800 on the first iteration
            index_start = year - y_start + 1600  # debug check: should be 225 on the first iteration

            if param[0] == 'residential_buildings':
                index_start = len(getattr(data, param[0])[param[1]]) - (
                        2100 - year - 1600) + 2  # debug check: should be 228 on the first iteration, and it is!

            print(f'Going through {category}')
            print(f'Case 3 transformation(__, {index_start}, {effect_percent})')

            data = getattr(data, param[0])[param[1]]
            sector = param[0]
            old = copy.deepcopy(data)

            # The parameters "people per dwellings" and "floor area per dwelling" are lists; thus, the len(data) does not
            # return the real length of the dataset. Therefore, we need to correct the end_index as follows.
            if sector == 'residential_buildings':
                end_index = len(data) + 3
            else:
                end_index = len(data)
            for y in range(index_start, end_index):
                # 1) Calculate the change between two consecutive years
                difference = old[y] - old[y - 1]
                # 2) Apply the effect to the change rate
                change = difference * (1 - effect_percent)
                # 3) Calculate the input data based on the data of the previous year and the calculated change
                data[y] = data[y - 1] + change
            print("iteration done")
        pass

    def mutateAlphaParameterised(self, start_leach, duration_alpha, end_leach_rate,
                                 sensitivityAnalysisCase="1A",
                                 start_re_introduction=[2080],
                                 duration_re_introduction=[10],
                                 end_re_introduction_rate=[0.5],
                                 ):  # Sensitivity Analysis case 5

        sensitivityAnalysisCase = sensitivityAnalysisCase.upper()
        sensitivityAnalysisCaseNumber = int(sensitivityAnalysisCase[0])
        sensitivityAnalysisCaseLetter = sensitivityAnalysisCase[1]

        alphaCases = {"1A": "single_ban_stock",
                      "1B": "single_ban_stock",
                      "2A": "double_ban_stock",
                      "2B": "double_ban_stock",
                      "3A": "single_ban_stock_inflow",
                      "4A": "double_ban_stock_inflow",
                      "5A": "unban_single_ban_stock_inflow",
                      "6A": "unban_double_ban_stock_inflow",
                      }
        alpha_case = alphaCases[sensitivityAnalysisCase]

        if sensitivityAnalysisCaseLetter.upper() == "A":
            categories = cfg.CASE5_A
        else:  # must be B otherwise
            categories = cfg.CASE5_B

        data = self.data

        for category in categories:
            param = category.split('.')
            attr = getattr(data, param[0])
            d: np.ndarray = attr[param[1]]

            splits = [i for i in data.residential_buildings['cohort_number']]  # e.g. [0, 1, 2, 3, 4, 5...]
            cohorts_starts = [i for i in data.residential_buildings['cohort_start']]
            cohorts_ends = [i for i in data.residential_buildings['cohort_end']]

            if param[0] == 'private_vehicles' or param[0] == 'public_vehicles' or param[0] == 'residential_buildings':
                types = [i for i in range(0, d.shape[2])]
            else:
                types = [0]

            for type in types:
                for split in splits:
                    print(f'Going through building type {type} cohort {split} of {category}')

                    input_data = self.data  # get the reference (and restore it from the previous iteration)

                    ## Run transformation of alpha ##
                    case_5_transformation(data=getattr(input_data, param[0])[param[1]],  # we restore `input_data`,
                                          start_leach=start_leach, cohorts_starts=cohorts_starts,
                                          cohorts_ends=cohorts_ends,
                                          split=split, type=type, alpha_case=alpha_case,
                                          duration_alpha=duration_alpha,
                                          end_leach_rate=end_leach_rate, sector=param[0])

                    ## Adjust other alpha related parameters ##
                    case_5_assign(input_data=input_data, sector=param[0], cohorts_starts=cohorts_starts,
                                  cohorts_ends=cohorts_ends, split=split, start_leach=start_leach,
                                  duration_alpha=duration_alpha, type=type, alpha_case=alpha_case)

                    if sensitivityAnalysisCaseNumber >= 3:  # starting from case 3 onwards, also transform the inflow
                        ## Modify Inflow ##
                        case_5_transformation_inflow(data=getattr(input_data, param[0])['typesplit'], type=type,
                                                     start_leach=start_leach, duration_alpha=duration_alpha,
                                                     alpha_case=alpha_case, end_leach_rate=end_leach_rate)

                    if sensitivityAnalysisCaseNumber >= 5:
                        ## Re-introduce the stock (e.g. ICEVs) via the inflow ##
                        case_5_re_introduction_inflow(data=getattr(input_data, param[0])['typesplit'], type=type,
                                                      start_re_introduction=start_re_introduction,
                                                      duration_re_introduction=duration_re_introduction,
                                                      alpha_case=alpha_case,
                                                      end_re_introduction_rate=end_re_introduction_rate,
                                                      sector=param[0])

        pass
