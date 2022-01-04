import matplotlib.pyplot as plt
from typing import List, Dict, Any

import utils


def show_plot(data: List[Dict[str, float]], result: Dict[str, Dict[str, Any]],
              forbidden_results=None):
    res_size = len(result)

    if res_size > 2:
        show_multiple_plots(result, forbidden_results)
    else:
        show_one_plot(data, result, forbidden_results)


def show_one_plot(data, result, forbidden_results):
    attribute, values = utils.get_attribute_and_values_as_list(data)

    plt.plot(attribute, values, '.')
    plt.title('Linear regression')
    plt.xlabel('mileage(km)')
    plt.ylabel('price')

    line_params = result[utils.KEY_LINE_PARAMS]
    plt.plot(attribute, [attr * line_params['theta1'] + line_params['theta0'] for attr in attribute],
             color='green', linewidth=3)

    if forbidden_results:
        line_params = forbidden_results[utils.KEY_LINE_PARAMS]
        plt.plot(attribute, [attr * line_params['theta1'] + line_params['theta0'] for attr in attribute],
                 color='red', linestyle=(0, (4, 10)), linewidth=2)
    plt.show()


def show_multiple_plots(result, forbidden_results):

    calculations_type = result[utils.KEY_DESCENT_CALCULATIONS]['type']
    data = result.pop(utils.KEY_FINISH_DATA)
    attribute = data['attribute']
    values = data['values']

    if calculations_type == '3d':
        show_multiple_plots_for_3d(result, attribute, values, forbidden_results)
    else:
        show_multiple_plots_for_2d(result, attribute, values, forbidden_results)


def show_multiple_plots_for_2d(result, attribute, values, forbidden_results):
    plt.figure(1)
    plt.plot(attribute, values, '.')
    plt.title('Linear regression')
    plt.xlabel('mileage(km)')
    plt.ylabel('price')

    line_params = result[utils.KEY_LINE_PARAMS]
    plt.plot(attribute, [attr * line_params['theta1'] + line_params['theta0'] for attr in attribute],
             color='green', linewidth=3)

    if forbidden_results:
        line_params = forbidden_results[utils.KEY_LINE_PARAMS]
        plt.plot(attribute, [attr * line_params['theta1'] + line_params['theta0'] for attr in attribute],
                 color='red', linestyle=(0, (4, 10)), linewidth=2)
    plt.show()

    calculations = result[utils.KEY_DESCENT_CALCULATIONS]
    plt.figure(2)
    plt.title('Losses graph')
    plt.ylabel('loss')
    plt.xlabel('mileage(km)')
    plt.plot(calculations['theta0_list'], calculations['losses'], '.')

    plt.show()


def show_multiple_plots_for_3d(result, attribute, values, forbidden_results):
    line_params = result[utils.KEY_LINE_PARAMS]

    plt.figure(1)
    plt.plot(attribute, values, '.')
    plt.title('Linear regression')
    plt.xlabel('mileage(km)')
    plt.ylabel('price')
    plt.plot(attribute, [attr * line_params['theta1'] + line_params['theta0'] for attr in attribute],
             color='green', linewidth=3)
    if forbidden_results:
        line_params = forbidden_results[utils.KEY_LINE_PARAMS]
        plt.plot(attribute, [attr * line_params['theta1'] + line_params['theta0'] for attr in attribute],
                 color='red', linestyle=(0, (4, 10)), linewidth=2)
    plt.show()

    calculations = result[utils.KEY_DESCENT_CALCULATIONS]
    losses = calculations['losses']
    theta0_list = calculations['theta0_list']
    theta1_list = calculations['theta1_list']

    # plt.figure(2)
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Losses from thetas')
    ax[0].plot(theta0_list, losses, '.')
    ax[1].plot(theta1_list, losses, '.')
    ax[0].set_title("theta0")
    ax[1].set_title("theta1")
    plt.ylabel("losses")
    plt.show()

    plt.figure(2)
    iterations = list(range(len(losses)))
    plt.plot(iterations, losses, '.')

    plt.title('Decrease of losses')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
