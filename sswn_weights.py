import numpy as np

from ss_mpc_hyperparameters import SS_MPC_PortfolioOptimisationParameters


class Weights:

    def __init__(self, structure):
        self.structure = structure
        self.rows_number = self.structure.state_space_size + self.structure.number_of_outputs + 2 * (
                           self.structure.number_of_inputs + self.structure.number_of_outputs +
                           self.structure.state_space_size)
        self.columns_number = self.structure.number_of_wavelons
        self.values = self.weight_init()
        params = SS_MPC_PortfolioOptimisationParameters()
        self.temperature = params.anneal_temperature

    def weight_init(self):

        return np.random.rand(*np.zeros((self.rows_number, self.columns_number), float).shape).dot(2)-1

    def anneal_all(self):

        melt_matrix = np.random.rand(*np.zeros((self.rows_number, self.columns_number), float).shape).dot(
            2*self.temperature)-self.temperature
        self.values = self.values + melt_matrix

    def anneal_single(self):
        return np.random.rand(*np.zeros((1, 1), float).shape).dot(2)-1
