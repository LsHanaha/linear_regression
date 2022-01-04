import numpy as np
from utils import get_attribute_and_values_as_list


def calculate_linear_regression_in_forbidden_way(data):
    attribute, values = get_attribute_and_values_as_list(data)
    slope, displacement = list(np.polyfit(attribute, values, 1))
    return {'line_params': {"theta1": slope, "theta0": displacement}}
