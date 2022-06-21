import numpy as np
import response as rsp
import md_wavelon as mdw


class Structure:

    """

    SSWN structure

    """

    def __init__(self, number_of_inputs, number_of_outputs, state_space_size, number_of_wavelons):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.state_space_size = state_space_size
        self.number_of_wavelons = number_of_wavelons
