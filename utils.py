from typing import List, Dict, Tuple


LEARNING_RATE = 0.5
THRESHOLD = 0.00001
MAX_EPOCH_COUNT = 5000
RESULT_FILE = 'result.json'

KEY_LINE_PARAMS = 'line_params'
KEY_DESCENT_CALCULATIONS = 'calculations'
KEY_FINISH_DATA = 'finish_data'


def count_mean(attribute_list: List[float]) -> float:
    size = len(attribute_list)
    attribute_sum = sum(attribute_list)
    return attribute_sum / size


def count_sum_of_squares(attribute_list, attribute_mean: float) -> float:
    return sum((x - attribute_mean) ** 2 for x in attribute_list)


def count_mean_square(attribute_list: List[float], attribute_mean: float) -> float:
    mean_square = (sum((val - attribute_mean) ** 2 for val in attribute_list)) ** 0.5
    return mean_square


def count_sum_of_diff(attribute_list: List[float], mean) -> float:
    return sum(value - mean for value in attribute_list)


def count_sum_of_mul_of_diff_for_two_lists(attribute_list1: List[float], mean1: float,
                                           attribute_list2: List[float], mean2: float) -> float:
    return sum((x - mean1) * (y - mean2) for x, y in zip(attribute_list1, attribute_list2))


def count_mse(attribute: List[float], values: List[float], line_params: Dict[str, float]) -> float:
    size = len(attribute)

    theta1 = line_params['theta1']
    theta0 = line_params['theta0']
    res = sum((y - (theta0 + theta1 * x)) ** 2 for x, y in zip(attribute, values)) / size
    return res


def count_mean_absolute_error(attribute: List[float], values: List[float],
                              line_params: Dict[str, float]) -> float:
    size = len(attribute)
    theta1 = line_params['theta1']
    theta0 = line_params['theta0']

    return sum((abs(((theta0 + theta1 * x) - y) / y) for x, y in zip(attribute, values))) / size


def count_r_square(attribute: List[float], values: List[float],
                   line_params: Dict[str, float]) -> float:

    size = len(attribute)
    mse = count_mse(attribute, values, line_params)

    value_mean = count_mean(values)
    return 1 - mse * size / sum((y - value_mean) ** 2 for y in values)


def count_theta0_derivative(theta0: float, theta1: float,
                            attribute: List[float], values: List[float],
                            learning_rate: float) \
        -> float:
    size = len(attribute)
    return sum((theta1 * x + theta0) - y for x, y in zip(attribute, values)) \
        * learning_rate / size


def count_theta1_derivative(theta0: float, theta1: float,
                            attribute: List[float], values: List[float],
                            learning_rate: float) \
        -> float:
    size = len(values)
    return sum(((theta1 * x + theta0) - y) * x for x, y in zip(attribute, values)) \
        * learning_rate / size


def get_attribute_and_values_as_list(data: List[Dict[str, float]]) \
        -> Tuple[List[float], List[float]]:

    attribute = []
    values = []

    for row in data:
        key, value = row.items()
        attribute.append(key[1])
        values.append(value[1])
    return attribute, values
