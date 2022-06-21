import numpy as np
import md_wavelon as mdw


class Response:

    def __init__(self, u_input, weight_matrix, structure):

        x = np.zeros((structure.state_space_size, u_input.shape[1]), float)
        y = np.zeros((structure.number_of_outputs, u_input.shape[1]), float)
        dt_size = structure.number_of_inputs + structure.number_of_outputs + structure.state_space_size
        d = weight_matrix.values[0:dt_size, ...]
        t = weight_matrix.values[dt_size:(dt_size+dt_size), ...]
        aj = np.zeros((structure.number_of_wavelons, 1), float)

        for column in range(u_input.shape[1]):

            if column == 0:
                x_input = np.zeros((structure.state_space_size, 1), float)
                y_input = np.zeros((structure.number_of_outputs, 1), float)
            else:
                x_input = x[..., column-1]
                y_input = y[..., column-1]

            zet = []
            zet.extend(y_input)
            zet.extend(x_input)
            zet.extend(u_input[..., column])

            # calculates the wavelons output for time relative to current column
            for j in range(structure.number_of_wavelons):
                aj[j] = mdw.multi_morlet(zet, d[..., j], t[..., j])

            # calculates the state space values for time relative to current column
            for i in range(structure.state_space_size):
                x[i, column] = weight_matrix.values[i + 2*dt_size, ...].dot(aj)

            # calculates the network output for time relative to current column
            for i in range(structure.number_of_outputs):
                y[i, column] = weight_matrix.values[i + structure.state_space_size + 2 * dt_size, ...].dot(aj)

        self.state = x
        self.output = y
        self.mse = None
