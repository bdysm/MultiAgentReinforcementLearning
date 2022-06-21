from portfolios import Portfolios
from ss_mpc_hyperparameters import SS_MPC_PortfolioOptimisationParameters
from xls_data_loader import LoadData
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def analyse_and_save(portfolios, input_data, num, filename):

    data_index = []
    current_assets_value = 1
    data_frame_header = ['']

    for asset in portfolios.assets:
        print(asset.name)
        data_frame_header.append(asset.name)
    data_frame_header.append('profit')


    my_np_panda = np.reshape(np.array(data_frame_header), [1,len(np.array(data_frame_header))])

    for x, list in enumerate(portfolios.history):

        assets = list[0]
        weights = list[1]
        data = list[2]
        gradient = []
        data_index.append(data.date())

        temp_panda = np.zeros([len(my_np_panda[0])])

        for a, asset in enumerate(assets):
            temp_panda[data_frame_header.index(asset.name)] = round(weights[a], 2)
            try:
                gradient.append(float(weights[a]) * asset.data.relative_data[asset.data.time_line.index(data) + 1])
            except:
                print('ok')

        current_assets_value = current_assets_value * (1 + sum(gradient))
        temp_panda[-1] = round(100 * current_assets_value, 2)
        my_np_panda = np.append(my_np_panda, [temp_panda], axis = 0)


    my_dataFrame = pd.DataFrame(data=my_np_panda[1:, 1:], index=data_index, columns=my_np_panda[0, 1:])
    print(my_dataFrame)

    book = load_workbook(filename)
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    sheet_name = 'Sheet' + str(num)
    my_dataFrame.to_excel(writer, sheet_name=sheet_name)
    writer.save()


def port_weights(num, params, filename):

    # Load the historical data
    input_data = LoadData('InputData.ods')

    # prepare the list of portfolios with all portfolio information ######################
    print('-----------------')
    print('Number of portfolios to be loaded: ' + str(len(params.portfolios_definition)))
    ######################################################################################
    portfolios = Portfolios(params, input_data.assets)
    portfolios.allocate_assets()  # creates the portfolios data
    portfolios.update_data_to_numpy()  # convert portfolio data into proper (numpy) format
    ######################################################################################

    max_horizon = portfolios.get_max_horizon()
    training_horizon = params.training_horizon
    prediction_steps = max_horizon - training_horizon + 1

    # Here we initialize the iteration over time
    for step in range(prediction_steps):

        # Time Step = step
        # Do all calculations for current time step with past data {training horizon}
        portfolios.do_all(step, training_horizon)

    analyse_and_save(portfolios, input_data, num, filename)


if __name__ == '__main__':

    # Load the analysis parameters {assumptions}
    parameters = SS_MPC_PortfolioOptimisationParameters()
    filename = 'Portfolio_opt_cases.xlsx'
    data = pd.ExcelFile(filename)

    port_weights(len(data.sheet_names), parameters, filename)

    """
    temp_portfolios_definition = parameters.portfolios_definition
    for i in range(len(parameters.portfolios_definition)):
        parameters.portfolios_definition = temp_portfolios_definition[i]
        

    parameters.portfolios_definition = temp_portfolios_definition
    port_weights(len(parameters.portfolios_definition), parameters)
    """