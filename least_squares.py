from typing import List, Dict
from utils import count_mean, get_attribute_and_values_as_list, count_sum_of_mul_of_diff_for_two_lists, \
    count_sum_of_squares, KEY_LINE_PARAMS


class LinearRegressionLeastSquares:

    def count(self, data: List[Dict[str, float]]):
        attribute, values = get_attribute_and_values_as_list(data)
        attribute_mean = count_mean(attribute)
        value_mean = count_mean(values)
        slope = self._count_slope(attribute, attribute_mean, values, value_mean)
        displacement = self._count_displacement(value_mean, attribute_mean, slope)
        return {KEY_LINE_PARAMS: {"theta1": slope, "theta0": displacement}}

    @staticmethod
    def _count_slope(attribute: list, attr_mean: float, values: list, value_mean: float) -> float:
        dividend = count_sum_of_mul_of_diff_for_two_lists(attribute, attr_mean, values, value_mean)
        divider = count_sum_of_squares(attribute, attr_mean)
        return dividend / divider

    @staticmethod
    def _count_displacement(y_mean, x_mean, slope) -> float:
        return y_mean - slope * x_mean
