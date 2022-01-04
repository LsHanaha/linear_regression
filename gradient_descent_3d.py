from typing import List, Dict
import utils


class LinearRegressionGradientDescent3D:

    def __init__(self):
        self._losses_list: List[float] = []
        self._theta0_list: List[float] = []
        self._theta1_list: List[float] = []

    def count(self, data: List[Dict[str, float]]):

        attribute, values = utils.get_attribute_and_values_as_list(data)

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

        attribute_norm, values_norm = self._normalize_data(attribute, values)
        result = self.count_thetas(attribute_norm, values_norm)

        theta0, theta1 = self._denormalize(result[utils.KEY_LINE_PARAMS], attribute, values)
        return {**result,
                utils.KEY_LINE_PARAMS: {"theta1": theta1, "theta0": theta0},
                utils.KEY_FINISH_DATA: {'attribute': attribute, 'values': values}}

    @staticmethod
    def _normalize_data(attribute, values):

        max_attribute = max(attribute)
        min_attribute = min(attribute)
        max_values = max(values)
        min_values = min(values)

        attr_normalized = [(x - min_attribute) / (max_attribute - min_attribute) for x in attribute]
        vals_normalized = [(y - min_values) / (max_values - min_values) for y in values]
        return attr_normalized, vals_normalized

    def count_thetas(self, attribute: List[float], values: List[float]):

        i = 0
        learning_rate = utils.LEARNING_RATE
        theta0 = 0
        theta1 = 0

        while i < utils.MAX_EPOCH_COUNT:
            dt0 = utils.count_theta0_derivative(theta0, theta1, attribute, values, learning_rate)
            dt1 = utils.count_theta1_derivative(theta0, theta1, attribute, values, learning_rate)

            theta0 -= dt0
            theta1 -= dt1
            loss = utils.count_mse(attribute, values, {"theta1": theta1, "theta0": theta0})

            learning_rate = self._reduce_learning_rate(loss, learning_rate)
            self._losses_list.append(loss)
            self._theta0_list.append(theta0)
            self._theta1_list.append(theta1)
            if self._check_threshold():
                print('Loss value less then threshold is reached!')
                break
            i += 1

        print('Learning done!')
        return {utils.KEY_LINE_PARAMS: {"theta1": theta1, "theta0": theta0},
                utils.KEY_DESCENT_CALCULATIONS: {'type': '3d', 'losses': self._losses_list,
                                                 'theta0_list': self._theta0_list,
                                                 'theta1_list': self._theta1_list}}

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
        if round(losses_mean, 9) - round(loss_last, 9) < utils.THRESHOLD / 10:
            return True
        return False

    @staticmethod
    def _denormalize(line_params, attribute, values):
        theta1, theta0 = line_params['theta1'], line_params['theta0']
        attr_params = [min(attribute), max(attribute)]
        vals_params = []
        for attr in attr_params:
            temp = theta1 * ((attr - min(attribute)) / (max(attribute) - min(attribute))) + theta0
            val = temp * (max(values) - min(values)) + min(values)
            vals_params.append(val)

        (x1, y1), (x2, y2) = zip(attr_params, vals_params)
        theta0 = (y2 * x1 - y1 * x2) / (x1 - x2)
        theta1 = (y1 - theta0) / x1
        return theta0, theta1
