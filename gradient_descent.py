from typing import Dict, List
import utils


class LinearRegressionGradientDescent:

    def __init__(self):
        self._losses_list: List[float] = []
        self._theta0_list: List[float] = []

    def count(self, data: List[Dict[str, float]]):

        attribute, values = utils.get_attribute_and_values_as_list(data)
        attr_mean = utils.count_mean(attribute)
        values_mean = utils.count_mean(values)

        while True:
            val = input('Вы можете ввести дополнительные значения, формат km:price, или что угодно '
                        'для продолжения\n')
            try:
                mileage, price = val.split(':')
                mileage, price = float(mileage), float(price)
                attribute.append(mileage)
                values.append(price)
            except (AttributeError, ValueError):
                break

        slope = self._count_slope(attribute, attr_mean, values, values_mean)
        result = self.count_theta0(attribute, values, slope)
        return {**result, utils.KEY_FINISH_DATA: {'attribute': attribute, 'values': values}}

    @staticmethod
    def _count_slope(attribute: list, attr_mean: float, values: list, values_mean: float) -> float:
        dividend = utils.count_sum_of_mul_of_diff_for_two_lists(attribute, attr_mean, values, values_mean)
        divider = utils.count_sum_of_squares(attribute, attr_mean)
        return dividend / divider

    def count_theta0(self, attribute: List[float], values: List[float], slope: float):

        i = 0
        learning_rate = utils.LEARNING_RATE
        theta0 = 0

        while i < utils.MAX_EPOCH_COUNT:
            dt0 = utils.count_theta0_derivative(theta0, slope, attribute, values, learning_rate)
            theta0 -= dt0
            loss = utils.count_mse(attribute, values, {"theta1": slope, "theta0": theta0})

            learning_rate = self._reduce_learning_rate(loss, learning_rate)
            self._losses_list.append(loss)
            self._theta0_list.append(theta0)
            if self._check_threshold():
                print('Loss value less then threshold is reached!')
                break
            i += 1

        print('Learning done!')
        return {utils.KEY_LINE_PARAMS: {"theta1": slope, "theta0": theta0},
                utils.KEY_DESCENT_CALCULATIONS: {'type': 'simple', 'losses': self._losses_list,
                                                 'theta0_list': self._theta0_list}}

    def _reduce_learning_rate(self, last_loss: float, learning_rate: float):
        if not len(self._losses_list):
            return learning_rate
        if last_loss > self._losses_list[-1]:
            learning_rate /= 2
        else:
            learning_rate *= 1.05
        return learning_rate

    def _check_threshold(self) -> bool:
        meaning_count = 10
        if len(self._losses_list) < meaning_count:
            return False
        losses_mean = utils.count_mean(self._losses_list[-meaning_count:])
        loss_last = self._losses_list[-1]
        if round(losses_mean, 9) - round(loss_last, 9) < utils.THRESHOLD:
            return True
        return False
