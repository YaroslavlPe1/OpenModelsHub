from sklearn.datasets import load_iris, load_digits
import pandas as pd

def load_sklearn_dataset(name):
    if name == 'iris':
        return load_iris(return_X_y=True)
    elif name == 'digits':
        return load_digits(return_X_y=True)
    else:
        raise ValueError(f"Dataset {name} not found.")

def load_csv_dataset(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]  # Все колонки, кроме последней — признаки
    y = data.iloc[:, -1]   # Последняя колонка — метки классов
    return X, y
