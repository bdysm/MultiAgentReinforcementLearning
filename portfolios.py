from assets import Assets
import numpy as np
from portfolio import Portfolio


class Portfolios:

    def __init__(self, params, assets):
        self.list = []  # list of portfolios
        self.number_of_portfolios = len(self.list)
        self.max_horizon = []
        self.assets = assets.assets_list
        self.params = params
        self.history = []

    def get_max_horizon(self):
        self.max_horizon = self.list[0].get_sample_length()
        return self.max_horizon

    def update_data_to_numpy(self):
        for portfolio in self.list:
            portfolio.prepare_portfolio_data()

    def allocate_assets(self):

        for single_portfolio_assets in self.params.portfolios_definition:
            portfolio_assets = Assets()

            for asset in self.assets:
                if asset.name in single_portfolio_assets:
                    portfolio_assets.append_asset(asset)

            single_portfolio = Portfolio(self.params, portfolio_assets)
            self.list.append(single_portfolio)

    def do_all(self, step, horizon):

        print('--------------------------------------------------------------------------')
        # print('Results are based only on historical data.')
        # print('SSWN prediction and MC prediction are not available yet.')
        # print('Here are optimum j_functions for each portfolio @ step number: ' + str(step))

        for portfolio in self.list:
            portfolio.solve_the_portfolio(step, horizon)

        best_portfolio = self.get_best_portfolio()
        temporary_portfolio = self.list[int(best_portfolio)]
        self.history.append([temporary_portfolio.assets,
                             temporary_portfolio.weights,
                             temporary_portfolio.data.quarter_date[horizon + step - 1]])
        print('Best portfolio is: ' + str(best_portfolio + 1))
        print('Time: ' + str(temporary_portfolio.data.quarter_date[horizon + step - 1]))
        print('asset & its weight in current portfolio:')

        for i in range(len(temporary_portfolio.assets)):
            temp_asset = temporary_portfolio.assets[i]
            print(temp_asset.name + ' ' + str(temporary_portfolio.weights[i]))

        # TODO: Rate the portfolios
        # TODO: Select optimal portfolio
        # TODO: Change the investment strategy if possible

    def get_best_portfolio(self):
        optimum = np.empty(shape=len(self.list))
        for i in range(len(self.list)):
            portfolio = self.list[i]
            optimum[i] = portfolio.optimum_return_ratio

        return np.argmax(optimum)
