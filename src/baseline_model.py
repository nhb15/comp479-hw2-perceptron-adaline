import random
import numpy as np
import argparse
import test_data_sets

def predictALabel(weights, values, threshold):
    sum = np.dot(values, weights, out=None)
    return 1 if sum > -threshold else 0

def main():
    random_and_no_update_weights = []

    '''
    LINEARLY SEPARATED DATA RUNS:
    '''
    values = test_data_sets.linearly_separated_data
    for _ in range(0, len(values[0])-1):
        random_and_no_update_weights.append(random.random())
    threshold = random.random()

    print(f"The base line weights for this linearly-separable training experiment are: {random_and_no_update_weights}")
    print(f"The base line threshold for this linearly-separable training experiment is: {threshold}")
    for tuple in test_data_sets.linearly_separated_data:
        print(f"The tuple: {tuple} is predictive of label {predictALabel(random_and_no_update_weights, tuple[0:2], threshold)}")

    '''
    NON-LINEARLY SEPARATED DATA RUNS:
    '''
    print(f"The base line weights for this non-linearly-separable training experiment are: {random_and_no_update_weights}")
    print(f"The base line threshold for this non-linearly-separable training experiment is: {threshold}")
    for tuple in test_data_sets.non_linearly_separated_data:
        print(f"The tuple: {tuple} is predictive of label {predictALabel(random_and_no_update_weights, tuple[0:2], threshold)}")
if __name__ == "__main__":
    main()
