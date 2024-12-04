from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def get_data(test_split=0.2, random_seed=0):
    housing = fetch_california_housing(data_home='./data')

    input = housing.data
    target = housing.target

    in_train, in_test, target_train, target_test = train_test_split(input, target, test_size=test_split, random_state=random_seed)

    return in_train, in_test, target_train, target_test

