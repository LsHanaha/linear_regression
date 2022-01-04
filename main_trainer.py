# The second program will be used to train your model. It will read your dataset file
# and perform a linear regression on the data.
# Once the linear regression has completed, you will save the variables theta0 and
# theta1 for use in the first program.

import argparse
import json
from typing import List, Dict

from read_data import read_data, show_residues
from analitics import LinearRegressionAnalytics
from least_squares import LinearRegressionLeastSquares
from gradient_descent import LinearRegressionGradientDescent
from gradient_descent_3d import LinearRegressionGradientDescent3D
from no_regrets import calculate_linear_regression_in_forbidden_way
from graphs import show_plot


from utils import RESULT_FILE, get_attribute_and_values_as_list, count_mean_absolute_error, \
    count_mse, count_r_square, KEY_LINE_PARAMS


def check_precision(start_data: List[Dict[str, float]], result: Dict[str, dict]) \
        -> None:

    attribute, values = get_attribute_and_values_as_list(start_data)
    line_params = result[KEY_LINE_PARAMS]

    mean_abs_error = count_mean_absolute_error(attribute, values, line_params)
    mse = count_mse(attribute, values, line_params)
    r_square = count_r_square(attribute, values, line_params)
    print(f"Got precision values\n - means absolute error: {mean_abs_error}\n - mean squared error: {mse}\n"
          f" - R square: {r_square}")


def store_result(result: Dict[str, float]):
    with open(RESULT_FILE, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def trainer(filename: str, calculation_type: str, graphs: bool, residues: bool, precision: bool,
            use_forbidden: bool):
    data = read_data(filename)
    if residues:
        show_residues(data)

    if calculation_type == 'gradient_descent':
        obj = LinearRegressionGradientDescent()
    elif calculation_type == 'gradient_descent_3d':
        obj = LinearRegressionGradientDescent3D()
    elif calculation_type == 'analytics':
        obj = LinearRegressionAnalytics()
    elif calculation_type == 'ordinary_least_squares':
        obj = LinearRegressionLeastSquares()
    else:
        raise

    result = obj.count(data)

    if use_forbidden:
        forbidden_results = calculate_linear_regression_in_forbidden_way(data)
    else:
        forbidden_results = None

    print(f"Got result {result[KEY_LINE_PARAMS]}")
    if use_forbidden:
        print(f"Forbidden results is {forbidden_results[KEY_LINE_PARAMS]}")

    if graphs:
        show_plot(data, result, forbidden_results)
    if precision:
        check_precision(data, result)

    store_result(result[KEY_LINE_PARAMS])
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-csv_file', help='Select csv file with cars mileage(km) and price(price).')
    parser.add_argument('-g', '--graphs', action='store_true', help='Show graphs')
    parser.add_argument('--type', default='gradient_descent', const='gradient_descent', nargs='?',
                        choices=('gradient_descent', 'gradient_descent_3d', 'analytics', 'ordinary_least_squares'),
                        help='Select type of calculating for linear regression. Default: %(default)s')
    parser.add_argument('-r', '--residues', help='Show graph of residues.', action='store_true')
    parser.add_argument('-p', '--precision', help='Count precision', action='store_true')
    parser.add_argument('-f', '--forbidden', help='Use forbidden np.polyfit for comparison.',
                        action='store_true')

    args = parser.parse_args()
    filename_ = args.csv_file
    calculation_type_ = args.type
    is_graphs_ = args.graphs
    is_residues_ = args.residues
    is_precision_ = args.precision
    use_forbidden_ = args.forbidden

    print(f"Selected options filename: {filename_ or 'default'}, show graphs: {is_graphs_}, "
          f"calculation type: {calculation_type_}, show residues: {is_residues_}, "
          f"count precision: {is_precision_}, forbidden: {use_forbidden_}")
    trainer(filename_, calculation_type_, is_graphs_,  is_residues_, is_precision_, use_forbidden_)
