
import pandas as pd

linearly_separated_data = [(1, 3, 0), (2, 2, 0), (3, 3, 0), (2, 4, 0), (1, 7, 0), (5, 3, 1), (6, 2, 1), (7, 1, 1), (8, 10, 1), (9, 0, 1)]

non_linearly_separated_data = [(1, 3, 1), (2, 2, 0), (3, 3, 1), (2, 4, 0), (1, 7, 1), (5, 3, 0), (6, 2, 1), (7, 1, 0), (8, 10, 1), (9, 0, 0)]

titanic_train_data = []
titanic_test_data = []

titanic_train_data_df = pd.read_csv('titanic/train.csv', delimiter=",")

titanic_test_data_df = pd.read_csv('titanic/test.csv', delimiter=",")



