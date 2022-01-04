from typing import List, Dict
from utils import count_mean, get_attribute_and_values_as_list, count_sum_of_mul_of_diff_for_two_lists, \
    count_sum_of_squares, KEY_LINE_PARAMS


class LinearRegressionAnalytics:

    def count(self, data: List[Dict[str, float]]):
        attribute, values = get_attribute_and_values_as_list(data)
        attribute_mean = count_mean(attribute)
        value_mean = count_mean(values)

        size = len(values)

        r = count_sum_of_mul_of_diff_for_two_lists(attribute, attribute_mean, values, value_mean)
        r = r / (count_sum_of_squares(attribute, attribute_mean) * count_sum_of_squares(values, value_mean)) ** 0.5
        s_y = (count_sum_of_squares(values, value_mean) / (size - 1)) ** 0.5
        s_x = (count_sum_of_squares(attribute, attribute_mean) / (size - 1)) ** 0.5

        slope = self._count_slope(r, s_x, s_y)
        displacement = self._count_displacement(value_mean, attribute_mean, slope)
        return {KEY_LINE_PARAMS: {"theta1": slope, "theta0": displacement}}

    @staticmethod
    def _count_slope(r, s_x, s_y):
        a = r * s_y / s_x
        return a

    @staticmethod
    def _count_displacement(y_mean, x_mean, slope):
        displacement = y_mean - slope * x_mean
        return displacement
