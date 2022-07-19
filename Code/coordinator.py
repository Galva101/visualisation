from ga.solution import Solution
from mfa.mfa_model import ModelMFA
from mfa.data_import import data_load_preparation
import copy
import numpy as np
import multiprocessing as mp
import _pickle as cPickle
import time
import bz2
from ga.fitness_evaluator import FitnessEvaluator
import os, sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Coordinator:
    """
    The Coordinator is keeping track of the population, initialises it,
    and executes the interaction between single solutions.
    It also stores a baseline MFA for reference.
    """

    data_baseline = ""
    result_baseline = ""

    populationSize = ""
    populationNumber = ""
    population = []

    populationFitnessHistory = []

    baselineEmissions = 0
    baselineEnergy = 0

    def __init__(self, populationSize):
        self.populationSize = populationSize
        self.populationNumber = 0

        print("Loading data into Coordinator")
        self.data_baseline = data_load_preparation()
        print("Starting Baseline MFA")
        self.result_baseline = ModelMFA(self.data_baseline).compilation_model()
        print("Baseline MFA finished \n")

    def initialisePopulation(self):
        # create a set of n solutions from the baseline and mutate them, then add them to the population List
        print("Initialising first Generation")

        # iterative version, the parallel one did not improve performance
        for i in range(self.populationSize):
            data_baseline_copy = copy.deepcopy(self.data_baseline)
            child_solution = Solution(data_baseline_copy, self.populationNumber, i)

            child_solution.initialiseChromosome()
            child_solution.mutateChromosome()
            with HiddenPrints():
                child_solution.applyChromosome()

            self.population.append(child_solution)
        print("Population initialisation Finished \n")

        f = FitnessEvaluator()
        self.baselineEmissions, self.baselineEnergy = f.extractFitnessValues(self.result_baseline)
        self.executePopulation()  # perform the initial generation's evaluation

    def executePopulation(self):
        # For every solution object in the coordinator's list, execute their MFA model
        # Parallel Version according to https://www.machinelearningplus.com/python/parallel-processing-python/
        cpuCount = min(mp.cpu_count(), 24)  # arbitrary upper bound
        pool = mp.Pool(cpuCount)
        result_objects = [pool.apply_async(s.runSolutionMFA, args=()) for s in self.population]
        results = [r.get() for r in result_objects]
        pool.close()
        pool.join()
        self.population = results

        print("Population " + str(self.populationNumber) + " Chromosomes executed")

    def evaluateGenerationFitness(self, fitnessOverAllYears):
        # For every solution object, depending on the scenario, execute the fitness evaluation
        f = FitnessEvaluator()
        generationFitness = []
        for solution in self.population:
            score = 0
            if fitnessOverAllYears:
                score = f.calculateFitnessMeanAllYears(solution.result, self.result_baseline)
            else:
                score = f.calculateFitnessFinalYear(solution.result, self.result_baseline)

            solution.fitnessValue = score
            generationFitness.append((score, solution.emissionValue, solution.energyValue, solution.chromosome))

        print("Population " + str(self.populationNumber) + " Fitness:")
        #For easier result collection, store all fitness values, sorted by highest first
        generationFitness = sorted(generationFitness, key=lambda tup: tup[0],
                                   reverse=True)  # https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index
        for i in range(len(generationFitness)):
            print(generationFitness[i][0], end="\t")
        print()
        self.populationFitnessHistory.append(generationFitness)  # add all individuals as a list of tuples
        pass

    def createNewPopulationFromCrossover(self):
        # create a new generation, and while it is smaller than the old one, draw two parents and merge them
        f = FitnessEvaluator()
        sortedGeneration = f.sortByFitnessAsc(self.population)
        parentSelectionForCrossover = f.linearRankSelection(sortedGeneration)

        newGeneration = []
        solutionNumber = 0
        while len(newGeneration) < self.populationSize:
            parent1 = np.random.choice(parentSelectionForCrossover)
            parent2 = np.random.choice(parentSelectionForCrossover)

            parent1Chromosome = parent1.chromosome
            parent2Chromosome = parent2.chromosome

            child1Chromosome, child2chromosome = parent1.crossover(parent1Chromosome, parent2Chromosome)

            if len(newGeneration) < self.populationSize:
                child1 = Solution(copy.deepcopy(self.data_baseline), self.populationNumber, solutionNumber)
                child1.chromosome = child1Chromosome
                newGeneration.append(child1)
                solutionNumber += 1

            if len(newGeneration) < self.populationSize: #in case we only need one more solution
                child2 = Solution(copy.deepcopy(self.data_baseline), self.populationNumber, solutionNumber)
                child2.chromosome = child2chromosome
                newGeneration.append(child2)
                solutionNumber += 1
        self.population = newGeneration

    def overwriteStartChromosome(self, chromosome):
        # not used in final implementation,
        # could be used to push the fitness by not starting over again completely
        for i in range(len(self.population)):
            if i < len(self.population) / 2:
                self.population[i].chromosome = chromosome

    def mutateGeneration(self):
        for solution in self.population:
            solution.mutateChromosome()  # draw the new value coefficients
            solution.applyChromosome()  # apply the new values to the data object in the solution
        pass

    def runSimulation(self, numberOfGenerations, storeInterval=5, fitnessOverAllYears=False, stoppingThreshold=-1):
        # run the simulation for a given number of generations
        print("\nSimulation Started\n")

        for i in range(numberOfGenerations):
            self.populationNumber += 1
            print("Running Generation " + str(self.populationNumber) + " of " + str(numberOfGenerations))
            self.evaluateGenerationFitness(fitnessOverAllYears)
            print("Generation " + str(self.populationNumber) + " Fitness Evaluation over")

            if self.getBestIndividualByReference().fitnessValue > stoppingThreshold > 0:  # if it is positive and the fitness is better
                print("\nSimulation Stopped at threshold\n")
                break

            self.createNewPopulationFromCrossover()
            print("Generation " + str(self.populationNumber) + " Offspring Created")

            with HiddenPrints():  # to silence all the prints from the simulations:
                # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
                self.mutateGeneration()
            print("Generation " + str(self.populationNumber) + " Offspring Mutated")

            self.executePopulation()
            print("Generation " + str(self.populationNumber) + " Offspring Executed\n")

            if i % storeInterval == 0 and not i <= 1:  # prevent save right at the start after the initalisation
                self.storePopulation()

        self.evaluateGenerationFitness(fitnessOverAllYears)
        self.storePopulation()  # final save

    def getBestIndividualByReference(self):
        # This is used in the plotting, so we can get the fittest solution by reference and pass it to these functions
        lastGeneration = self.populationFitnessHistory[-1]
        bestIndividualTuple = lastGeneration[0]
        bestIndividualFitness = bestIndividualTuple[0]

        bestIndividual = None
        for individual in self.population:
            if individual.fitnessValue == bestIndividualFitness:
                bestIndividual = individual

        return bestIndividual

    def getBestIndividualOfAllGenerations(self):
        bestIndividual = ""
        bestFitness = 0
        index = 0
        for i in range(len(self.populationFitnessHistory)):
            gen = self.populationFitnessHistory[i]
            bestIndividualTuple = gen[0]
            bestIndividualFitness = bestIndividualTuple[0]

            if bestIndividualFitness > bestFitness:
                bestIndividual = bestIndividualTuple
                bestFitness = bestIndividualFitness
                index = i

        print("\nBest Individual of all generations:")
        print("Index: " + str(index) + ", " + str(bestFitness) + ", " + str(bestIndividual))

    def calculateGenerationDiversity(self, generation):
        #using the diversity formula, calculate it for the passed generation
        fitnessDict = {}
        for individual in generation:
            fitnessValue = individual[0]
            if fitnessValue in fitnessDict:
                fitnessDict[fitnessValue] += 1
            else:
                fitnessDict[fitnessValue] = 1

        n = len(generation)
        s = 0
        for species in fitnessDict:
            p = fitnessDict[species] / n
            s += p * np.log(p)
        return -s

    def calculateGenerationEveness(self, generation):
        # using the evenness formula and the diversity, calculate it for the passed generation
        diversity = self.calculateGenerationDiversity(generation)

        fitnessDict = {}
        for individual in generation:
            fitnessValue = individual[0]
            if fitnessValue in fitnessDict:
                fitnessDict[fitnessValue] += 1
            else:
                fitnessDict[fitnessValue] = 1
        n = len(generation)

        return diversity / np.log(n)

    def printPopulationFitnessHistory(self):
        print("\nPopulation Fitness History:")
        for l in self.populationFitnessHistory:
            for i in range(len(l)):
                print(l[i][0], end="\t")
            print("")
        pass

    def printBestNofEachGenerationAsCSV(self, n=5):
        # this is used in the rank plot (the blue one), where for each generation the best n (e.g. 5) are printed as CSV
        print("\nBest Fitness of n=" + str(n) + " Individuals per Generation for R as CSV:")
        print("\"Generation\", \"Position\", \"Fitness\"")
        s = ""
        for g in range(len(self.populationFitnessHistory)):
            generation = self.populationFitnessHistory[g]
            for i in range(len(generation)):
                if i < n:
                    individual = generation[i]
                    s += str(g) + ", " + str(i) + ", " + str(individual[0]) + "\n"
        print(s)
        pass

    def printDiversityAndEvenness(self):
        # prints the diversity and evenness of every generation as CSV
        print("\nDiversity and Eveness of each Generation for R as CSV:")
        print("\"Generation\", \"Diversity\", \"Evenness\", \"PopulationSize\"")
        s = ""
        for g in range(len(self.populationFitnessHistory)):
            generation = self.populationFitnessHistory[g]
            diversity = self.calculateGenerationDiversity(generation)
            evenness = self.calculateGenerationEveness(generation)
            populationSize = self.populationSize
            s += str(g) + ", " + str(diversity) + ", " + str(evenness) + ", " + str(populationSize) + "\n"
        print(s)
        pass

    def printFitnessTimeSeries(self):
        # prints the year-wise fitness improvement as an r "vector" for direct use in an R markdown file for plots
        print("\nBest Individual's Fitness over all years compared to the reference for R as a Vector:")
        bestIndividual = self.getBestIndividualByReference()
        f = FitnessEvaluator()
        fitnessTimeSeries = f.calculateFitnessTimeSeries(bestIndividual.result, self.result_baseline)

        res = ""
        for f in fitnessTimeSeries:
            res += str(f) + ", "
        print("series <- c(" + res[:-2] + ")")

    def printMaxFitnessPerGeneration(self):
        print("\nMax Fitness per Generation:")
        for gen in self.populationFitnessHistory:
            individualTuple = gen[0]
            print(individualTuple)

    def printPercentagesPerFitnessOfAllGenerations(self, lastGenOnly=False):
        # prints all 5 percentage values for each generation, to be used as a CSV in R
        print("\nPercentage Numbers of the Best Individual of every Generation for R as CSV:")

        print("\"Fitness\", \"Generation\", \"Variable\", \"Value\"")
        for i in range(len(self.populationFitnessHistory)):
            if not lastGenOnly or (lastGenOnly and i == len(self.populationFitnessHistory) - 1):
                # this prints always if lastGen is false, or if it is true only on the last generation iteration

                generation = self.populationFitnessHistory[i]
                individualTuple = generation[0]
                fitness = individualTuple[0]

                population = individualTuple[3][0]
                renovationCycle = individualTuple[3][2]
                energyIntensity = individualTuple[3][4]
                typesplitSplitshare = individualTuple[3][6]
                dwelling = individualTuple[3][8]

                print(str(fitness) + ", " + str(i) + ", Population, " + str(population))
                print(str(fitness) + ", " + str(i) + ", Renovation Cycle, " + str(renovationCycle))
                print(str(fitness) + ", " + str(i) + ", Energy Intensity, " + str(energyIntensity))
                print(str(fitness) + ", " + str(i) + ", Typesplit and Splitshare, " + str(typesplitSplitshare))
                print(str(fitness) + ", " + str(i) + ", Dwelling, " + str(dwelling))

        pass

    def printYearsPerFitnessOfALlGenerations(self, lastGenOnly=False):
        # prints all 5 year values for each generation, to be used as a CSV in R
        print("\nYears of the Best Individual of every Generation for R as CSV:")

        print("\"Fitness\", \"Generation\", \"Variable\", \"Value\"")
        for i in range(len(self.populationFitnessHistory)):
            if not lastGenOnly or (lastGenOnly and i == len(self.populationFitnessHistory) - 1):
                generation = self.populationFitnessHistory[i]
                individualTuple = generation[0]
                fitness = individualTuple[0]

                population = individualTuple[3][1]
                renovationCycle = individualTuple[3][3]
                energyIntensity = individualTuple[3][5]
                typesplitSplitshare = individualTuple[3][7]
                dwelling = individualTuple[3][9]

                print(str(fitness) + ", " + str(i) + ", Population, " + str(population))
                print(str(fitness) + ", " + str(i) + ", Renovation Cycle, " + str(renovationCycle))
                print(str(fitness) + ", " + str(i) + ", Energy Intensity, " + str(energyIntensity))
                print(str(fitness) + ", " + str(i) + ", Typesplit and Splitshare, " + str(typesplitSplitshare))
                print(str(fitness) + ", " + str(i) + ", Dwelling, " + str(dwelling))
        pass

    def printGenerationValuesForRegression(self, numGen, lastGenOnly=False, bestIndividualOnly=False):
        # prints all fitness values, as well as diversity and evenness to be used as a CSV in R
        print("\nThe Regression values for R as a CSV:")

        print("\"Fitness\", \"Emission\", \"Energy\",\"diversity\", \"evenness\",\"populationSize\", "
              "\"numberOfGenerations\"")

        for i in range(len(self.populationFitnessHistory)):
            if not lastGenOnly or (lastGenOnly and i == len(self.populationFitnessHistory) - 1):
                gen = self.populationFitnessHistory[i]

                for j in range(len(gen)):
                    if not bestIndividualOnly or (bestIndividualOnly and j == 0):
                        individualTuple = gen[j]

                        fitness = individualTuple[0]
                        emission = individualTuple[1]
                        energy = individualTuple[2]

                        diversity = self.calculateGenerationDiversity(gen)
                        evenness = self.calculateGenerationEveness(gen)

                        populationSize = self.populationSize
                        numberOfGenerations = numGen

                        print(str(fitness) + ", " + str(emission) + ", " + str(energy) + ", "
                              + str(diversity) + ", " + str(evenness) + ", "
                              + str(populationSize) + ", " + str(numberOfGenerations))
        pass

    def printPercentagesPerFitnessOfCompletePopulation(self):
        # prints all 5 percent values for the final generation, to be used as a CSV in R
        # Might actually be a duplicate with the one above that can also print only the final generation
        print("\nPercentage Numbers of every single Individual of the Generation for R as CSV:")

        print("\"Fitness\", \"Individual\", \"Variable\", \"Value\"")
        lastGen = self.populationFitnessHistory[-1]
        for i in range(len(lastGen)):
            individualTuple = lastGen[i]

            fitness = individualTuple[0]
            population = individualTuple[3][0]
            renovationCycle = individualTuple[3][2]
            energyIntensity = individualTuple[3][4]
            typesplitSplitshare = individualTuple[3][6]
            dwelling = individualTuple[3][8]

            print(str(fitness) + ", " + str(i) + ", Population, " + str(population))
            print(str(fitness) + ", " + str(i) + ", Renovation Cycle, " + str(renovationCycle))
            print(str(fitness) + ", " + str(i) + ", Energy Intensity, " + str(energyIntensity))
            print(str(fitness) + ", " + str(i) + ", Typesplit and Splitshare, " + str(typesplitSplitshare))
            print(str(fitness) + ", " + str(i) + ", Dwelling, " + str(dwelling))
        pass

    def printYearsPerFitnessOfCompletePopulation(self):
        # prints all 5 year values for the final generation, to be used as a CSV in R
        # Might actually be a duplicate with the one above that can also print only the final generation
        print("\nYears of every single Individual of the Generation for R as CSV:")

        print("\"Fitness\", \"Individual\", \"Variable\", \"Value\"")
        lastGen = self.populationFitnessHistory[-1]
        for i in range(len(lastGen)):
            individualTuple = lastGen[i]

            fitness = individualTuple[0]
            population = individualTuple[3][1]
            renovationCycle = individualTuple[3][3]
            energyIntensity = individualTuple[3][5]
            typesplitSplitshare = individualTuple[3][7]
            dwelling = individualTuple[3][9]

            print(str(fitness) + ", " + str(i) + ", Population, " + str(population))
            print(str(fitness) + ", " + str(i) + ", Renovation Cycle, " + str(renovationCycle))
            print(str(fitness) + ", " + str(i) + ", Energy Intensity, " + str(energyIntensity))
            print(str(fitness) + ", " + str(i) + ", Typesplit and Splitshare, " + str(typesplitSplitshare))
            print(str(fitness) + ", " + str(i) + ", Dwelling, " + str(dwelling))
        pass

    def findBestFitnessAndChromosome(self): # unused
        bestSolution = self.populationFitnessHistory[-1]
        return bestSolution[0][0], bestSolution[0][1]

    def printBestFitnessForR(self):
        print("\nBest Fitness per Generation for R as a Vector:")
        res = ""
        for f in self.populationFitnessHistory:
            res += str(f[0][0]) + ", "
        print("f <- c(" + res[:-2] + ")")

    def storePopulation(self):
        # Stores the population list as a pickle file to be loaded again if progress should be saved
        try:
            print("Starting Generation " + str(self.populationNumber) + " Save")
            start_running_time = time.time()
            storageDictionary = {
                "populationSize": self.populationSize,
                "populationNumber": self.populationNumber,
                "population": self.population,
                "result_baseline": self.result_baseline,
                "data_baseline": self.data_baseline,
                "populationFitnessHistory": self.populationFitnessHistory,
            }

            # old fast version
            # pickle.dump(storageDictionary, open("./ga/storedGeneration.pickle", "wb"))

            with bz2.BZ2File("./ga/storedGeneration.pickle.pbz2", "w") as f:  # compression according to
                # https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
                cPickle.dump(storageDictionary, f)

            print("Generation " + str(self.populationNumber) + " Stored")
            print("--- %s seconds ---" % (time.time() - start_running_time))
        except Exception as e:
            print("Error while storing the Generation:", e)

    def loadPopulation(self):
        # counterpart to the store function
        try:
            print("Starting Generation Loading")
            start_running_time = time.time()

            # old, fast version
            # storageDictionary = pickle.load(open("./ga/storedGeneration.pickle", "rb"))

            data = bz2.BZ2File("./ga/storedGeneration.pickle.pbz2", "rb")
            storageDictionary = cPickle.load(data)

            self.population = storageDictionary["population"]
            self.populationSize = storageDictionary["populationSize"]
            self.populationNumber = storageDictionary["populationNumber"]
            self.result_baseline = storageDictionary["result_baseline"]
            self.data_baseline = storageDictionary["data_baseline"]
            self.populationFitnessHistory = storageDictionary["populationFitnessHistory"]
            print("Generation Loaded")
            print("--- %s seconds ---" % (time.time() - start_running_time))
        except Exception as e:
            print("Error while loading the Generation:", e)

    def executeMassBalanceTest(self):
        for child_solution in self.population:
            child_solution.massBalanceTest()
