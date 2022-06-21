import math
import numpy as np

from population import Population
from ss_mpc_hyperparameters import SS_MPC_PortfolioOptimisationParameters
from sswn_structure import Structure


class ModelForPortfolioRHC:

    def __init__(self, data, params):
        self.structure = Structure(params.input_size, params.output_size,
                                   params.state_size, params.wavelons_number)
        self.data = data
        self.weights = self.compute_weights(params.pop_size, params.best_size,
                                            params.random_size, params.anneal_size,
                                            params.sswn_stop_condition, params.sswn_max_count)

    def compute_weights(self, pop_size, best_size, random_size, anneal_size, stop_cond, max_count):
        population = Population(self.structure, pop_size)
        population.compute(self.structure, self.data.u_input, self.data.y_target)
        params = SS_MPC_PortfolioOptimisationParameters()
        count = 0
        while population.get_best_mse() > stop_cond and count < max_count:
            if population.size > 1:
                population.update(best_size, random_size, anneal_size, self.structure)
                population.compute(self.structure, self.data.u_input, self.data.y_target)
                # print(str(population.get_best_mse()) + '   ' + str(count))
                count += 1
            else:
                temperature = params.anneal_temperature / (1 + math.log(count + 1))
                for row in range(population.genotyp[0].weight.rows_number):
                    for col in range(population.genotyp[0].weight.columns_number):
                        previous_value = np.empty(1)
                        previous_value[0] = population.genotyp[0].weight.values[row, col]
                        previous_mse = np.empty(1)
                        previous_mse[0] = population.genotyp[0].mse
                        population.anneal(self.structure, row, col)
                        population.compute(self.structure, self.data.u_input, self.data.y_target)
                        if population.genotyp[0].mse > previous_mse[0]:
                            delta = population.genotyp[0].mse - previous_mse[0]
                            condition = math.exp(-delta/temperature)
                            if np.random.rand(1) > condition:
                                population.genotyp[0].weight.values[row, col] = previous_value[0]
                print(str(count) + ':   ' + str(population.get_best_mse()))
                count += 1

        return population.get_best_genotyp().weight


class MPCDataForTraining:

    def __init__(self):
        self.u_input = None
        self.y_target = None
