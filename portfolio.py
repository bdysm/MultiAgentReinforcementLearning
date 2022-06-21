import numpy as np
from model_for_mpc import MPCDataForTraining, ModelForMPC
from response import Response


class Portfolio:

    def __init__(self, params, assets):
        self.params = params
        self.assets = assets.assets_list
        self.data = PortfolioData()
        self.constraints = params.portfolio_constraints
        self.short_data = None
        self.weights = None  # self.init_portfolio_weights()
        # self.weighted_data = None  # self.get_weighted_data()
        self.model = None  # ModelForMPC(self.weighted_data, self.params)
        self.optimum_return_ratio = None

    def init_portfolio_weights(self):
        inp = self.params.portfolio_number_of_assets
        weights = np.ndarray(shape=inp)
        return weights

    def get_weighted_return(self):
        weighted_return = np.empty(shape=self.params.training_horizon)
        for i in range(len(weighted_return)):
            weighted_return[i] = np.sum(self.short_data.relative[:, i] * self.weights)
        return weighted_return

    @staticmethod
    def get_derivative(weighted_return):
        derivative = np.empty(shape=len(weighted_return)-1)
        for i in range(len(weighted_return) - 1):
            derivative[i] = weighted_return[i+1] - weighted_return[i]
        return derivative

    def prepare_portfolio_data(self):
        for asset in self.assets:
            self.data.append(asset)
        self.data.to_numpy()

    def get_sample_length(self):
        return len(self.data.quarter_date)

    def solve_the_portfolio(self, step, horizon):
        # prepare the data for limited samples:
        self.short_data = PortfolioShortData(self.data, [step, step + horizon])
        assets_respective_volatility = self.do_statistics()
        average_return_vector = self.short_data.relative.mean(1)*4  # TODO: why multiplied by 4 ??

        # optimization of portfolio weights
        self.get_optimum_portfolio_weights(assets_respective_volatility, average_return_vector)

        self.optimum_return_ratio = self.calculate_current_j_function_value(assets_respective_volatility,
                                                                            average_return_vector)

        # TODO: as a return from above we get weighted portfolio VOLATILITY and optimal portfolio weights {pw}

        # -------------------------------------------------------------------------------------
        # TODO: prediction for one step forward {sswn} and long horizon prediction {Monte Carlo}
        mpc_data = MPCDataForTraining()
        mpc_data.u_input = np.empty(shape=(self.params.input_size, horizon - 1))
        mpc_data.u_input[0:self.params.portfolio_number_of_assets, :] = self.short_data.sswn_data[:, 0:horizon - 1]
        mpc_data.u_input[-1, 0] = 0
        weighted_return = self.get_weighted_return()
        mpc_data.y_target = np.empty(shape=(self.params.output_size, horizon - 1))
        mpc_data.y_target[:] = self.get_derivative(weighted_return)
        mpc_data.u_input[-1, 1:] = mpc_data.y_target[:, 0:-1]
        # TODO: check if mpc_data are prepared correctly !!!
        self.model = ModelForMPC(mpc_data, self.params)
        if self.params.sswn_based:
            prediction_data_u_input = np.empty(shape=(self.params.input_size, horizon))
            prediction_data_u_input[0:self.params.portfolio_number_of_assets, :] = self.short_data.sswn_data[:, 0:horizon]
            prediction_data_u_input[-1, 1:] = mpc_data.y_target[:, :]
            prediction = Response(prediction_data_u_input, self.model.weights, self.model.structure)
            predicted = prediction.output[:, -1]*self.short_data.relative[-1, -1] + self.short_data.relative[-1, -1]
            self.optimum_return_ratio = self.calculate_current_j_function_value(assets_respective_volatility,
                                                                                predicted)
            print('optimum SSWN based E{Rp}/s: ' + str(self.optimum_return_ratio))
        else:
            print('optimum history based E{Rp}/s: ' + str(self.optimum_return_ratio))
        print('###############################################')

        # TODO: predict the next step expected return using Response(mpc_data.input, sswn_weights_matrix, structure)
        # TODO: compute the Monte Carlo prediction

    def do_statistics(self):
        correlation_matrix = np.corrcoef(self.short_data.relative)  # why not origin?
        vv = np.empty(0)
        for row in self.short_data.relative:
            vv = np.append(vv, np.std(row, ddof=1)*2)

        assets_respective_volatility = (vv * correlation_matrix).transpose() * vv
        return assets_respective_volatility

    def calculate_current_j_function_value(self, respective_volatility, average_return_vector):
        # calculate weighted portfolio VOLATILITY
        weighted_volatility = self.get_weighted_volatility(respective_volatility)
        # calculate weighted expected return
        weighted_return = sum(average_return_vector * self.weights)
        # calculate J-function
        current_j_function_value = weighted_return/weighted_volatility

        return current_j_function_value

    def get_optimum_portfolio_weights(self, assets_respective_volatility, average_return_vector):
        j_delta = np.empty([self.params.portfolio_number_of_assets])
        for i in range(self.params.portfolio_number_of_assets):
            temporary_weights = np.empty([self.params.portfolio_number_of_assets])
            temporary_weights[:] = self.params.initial_weights[:]
            temporary_weights[:] = temporary_weights[:] - 0.001
            temporary_weights[i] = temporary_weights[i] + 0.001 * (self.params.portfolio_number_of_assets + 1)
            self.weights = temporary_weights
            j_delta[i] = self.calculate_current_j_function_value(assets_respective_volatility, average_return_vector)

        j_delta_sorted = np.sort(j_delta)

        for i in range(self.params.portfolio_number_of_assets):
            for j in range(self.params.portfolio_number_of_assets):
                if j_delta[i] == j_delta_sorted[j]:
                    self.weights[i] = self.params.vector[j]

    def get_weighted_volatility(self, respective_volatility):
        weighted_volatility = np.sqrt(np.sum((self.weights * respective_volatility).transpose() * self.weights))
        return weighted_volatility


class PortfolioData:

    def __init__(self):
        self.quarter_date = []
        self.origin = []
        self.relative = []
        self.sswn_data = []

    def append(self, asset):
        if len(self.quarter_date) == 0:
            self.quarter_date = asset.data.time_line

        self.origin.append(asset.data.origin_data)
        self.relative.append(asset.data.relative_data)
        self.sswn_data.append(asset.data.sswn_data)

    def to_numpy(self):
        self.origin = np.array(self.origin)
        self.relative = np.array(self.relative)
        self.sswn_data = np.array(self.sswn_data)


class PortfolioShortData:

    def __init__(self, data, time_range):
        self.quarter_date = data.quarter_date[time_range[0]:time_range[1]]
        self.origin = data.origin[:, time_range[0]:time_range[1]]
        self.relative = data.relative[:, time_range[0]:time_range[1]]
        self.sswn_data = data.sswn_data[:, time_range[0]:time_range[1]]
