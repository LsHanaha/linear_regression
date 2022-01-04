# The first program will be used to predict the price of a car for a given mileage.
# When you launch the program, it should prompt you for a mileage, and then give
# you back the estimated price for that mileage.

import json
import os
import utils
import argparse
from typing import Optional, Dict

from no_regrets import calculate_linear_regression_in_forbidden_way
from read_data import read_data


def main(use_forbidden: bool):

    theta0, theta1 = get_thetas()

    forbidden = None
    if use_forbidden:
        forbidden = calculate_linear_regression_in_forbidden_way(read_data(None))
        forbidden = forbidden[utils.KEY_LINE_PARAMS]

    while True:
        mileage = input('\nFor price prediction just input mileage of car. '
                        'To stop the program send "stop"\nmiles = ')

        if mileage.lower() == 'stop':
            print('Stopped')
            break
        try:
            mileage = float(mileage)
        except (ValueError, TypeError):
            print("Wrong type for mileage. Expected float, got not float")
            continue
        predict_price(mileage, theta1, theta0, forbidden)


def get_thetas():

    path = os.path.join(os.getcwd(), utils.RESULT_FILE)
    if not os.path.exists(path):
        print("Thetas is not defined. Sorry, i'm leaving")
        exit(0)

    with open(path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("Result format is broken. Exit now")
        exit(1)

    theta0 = data.get('theta0')
    theta1 = data.get('theta1')
    if theta0 is None or theta1 is None:
        print("One of theta is not defined. Exit now")
        exit(1)

    try:
        theta0 = float(theta0)
        theta1 = float(theta1)
    except (TypeError, ValueError):
        print('Could not convert one of theta to float. Exit now')
    return theta0, theta1


def predict_price(mileage: float, theta1: float, theta0: float,
                  forbidden: Optional[Dict[str, float]]):

    predicted = theta1 * mileage + theta0
    if predicted <= 0:
        print('Price of your car is going to be equal zero. Just move it to recycle')
    else:
        print(f"Price for mileage {mileage} is going to be near {round(predicted, 2)}.")

    if forbidden:
        theta0 = forbidden['theta0']
        theta1 = forbidden['theta1']
        predicted = theta1 * mileage + theta0
        print(f"Forbidden calculation. Price for mileage {mileage} is "
              f"going to be near {round(predicted, 2)}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--forbidden', help='Use forbidden np.polyfit for comparison.',
                        action='store_true')

    args = parser.parse_args()

    main(args.forbidden)
