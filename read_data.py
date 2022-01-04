import csv
import os
import matplotlib.pyplot as plt
from typing import List, Dict

from utils import count_mean, get_attribute_and_values_as_list


_DEFAULT_FILE = 'data.csv'


def read_data(filename) -> List[Dict[str, float]]:

    filename = filename or _DEFAULT_FILE

    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {filename} in {os.getcwd()} not found!")
    if not path.endswith('.csv'):
        raise TypeError(f"File {filename} not a .csv file!")
    csv_data = []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        try:
            header = next(csvreader)
        except StopIteration:
            raise IndexError(f"File {filename} is empty!")
        for row in csvreader:
            if len(row) != 2:
                raise IndexError(f"Invalid row in csv file {row}. Should be 2 columns: km and price!")
            named_row = zip(header, [float(val) for val in row])
            csv_data.append(dict(named_row))
    if not csv_data:
        raise IndexError(f"File {filename} is empty!")
    if len(csv_data) < 2:
        raise IndexError(f"Too few values. At least 2!")
    return csv_data


def show_residues(data: List[Dict[str, float]]) -> None:

    attribute, values = get_attribute_and_values_as_list(data)

    y_mean = count_mean(values)
    values = [val - y_mean for val in values]

    plt.plot(attribute, values, '.')
    plt.title('График остатков')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()
    res = input("Похожа ли зависимость на линейную? Отсутствуют ли выбросы?\nY\\N\n")
    if res.lower() != 'y':
        print('Жаль. Введите более корректные данные.')
        exit(0)

    return


if __name__ == '__main__':
    from pprint import pp
    data_ = read_data(None)
    show_residues(data_)
    pp(data_)
