import numpy as np


class SS_MPC_PortfolioOptimisationParameters:

    def __init__(self):

        self.input_size = 7
        self.output_size = 1
        self.state_size = 12
        self.wavelons_number = 12

        self.pop_size = 3
        self.best_size = 1
        self.random_size = 1
        self.anneal_size = self.pop_size - self.best_size - self.random_size
        self.anneal_temperature = 0.001
        self.sswn_based = False

        self.sswn_stop_condition = 0.001

        if self.sswn_based:
            self.sswn_max_count = 300
        else:
            self.sswn_max_count = 0

        self.portfolio_constraints = [0.05, 0.35]
        self.training_horizon = 21
        self.portfolio_number_of_assets = 6
        self.initial_weights = self.initialize_portfolio_weights()
        self.vector = self.get_vector()
        p001 = ['Gold',   'Copper', 'SnP', 'Swiss', 'Singapore', 'WIG20']
        p002 = ['Silver', 'Copper', 'DAX', 'Swiss', 'Singapore', 'WIG20']
        p003 = ['Gold',   'Copper', 'DAX', 'Swiss', 'Singapore', 'Bovespa']
        p004 = ['Gold',   'Palladium', 'DAX', 'Kospi', 'Singapore', 'Bovespa']
        p005 = ['Silver', 'Zinc', 'Kospi', 'Swiss', 'Singapore', 'Bovespa']
        p006 = ['Palladium', 'Zinc', 'SnP', 'Kospi', 'Swiss', 'WIG20']
        p007 = ['Gold', 'Copper', 'SnP', 'Swiss', 'Singapore', 'Bovespa']
        p008 = ['Silver', 'Copper', 'SnP', 'Swiss', 'Singapore', 'WIG20']
        p009 = ['Gold', 'Copper', 'DAX', 'SnP', 'Singapore', 'Bovespa']
        p010 = ['Gold', 'SnP', 'Silver', 'DAX', 'Palladium', 'Kospi']
        self.portfolios_definition = [p001, p002, p003, p010]


    def initialize_portfolio_weights(self):

        weights =np.empty([self.portfolio_number_of_assets])
        weights[:] = 1/self.portfolio_number_of_assets
        return weights

    def get_vector(self):
        num = self.portfolio_number_of_assets
        vector = np.empty(shape=num)
        min_value = self.portfolio_constraints[0]
        max_value = self.portfolio_constraints[1]
        sum_value = 1
        for i in range(num):
            till_end = num - i
            value_left = sum_value - np.sum(vector[0:i])
            if i == 0:
                vector[i] = min_value
            elif i == 1:
                vector[i] = max_value
            else:
                if np.sum(vector[0:i - 1]) < sum_value:
                    if value_left > (till_end * min_value + max_value):
                        vector[i] = max_value
                    elif i == (num - 1):
                        vector[i] = value_left
                    else:
                        vector[i] = min_value
        vector = np.sort(vector)

        return vector
