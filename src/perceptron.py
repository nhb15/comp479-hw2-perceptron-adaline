import random
import numpy as np
import argparse

def predictALabel(weights, values, threshold):
    sum = np.dot(values, weights, out=None)
    return 1 if sum > threshold else 0

def train(learning_rate, num_passes, threshold, values):
    num_data_points = len(values)
    num_dimensions = len(values[0]) - 1
    print(num_dimensions)

    weights = []
    features = []
    for dimension in range (0, num_dimensions):
        features.append([])
        for feature in range(0, num_data_points):
            features[dimension].append(values[feature][dimension])
        weights.append(random.random())

    for _ in range(num_passes):
        for index in range(num_data_points):
            prediction_for_row = predictALabel(weights, [data[index] for data in features], threshold)
            for weight_index, _ in enumerate(weights):
                print(values[index])
                weights[weight_index] = learning_rate * (values[index][len(values[index]) - 1] - prediction_for_row) * values[index][weight_index]
            threshold = learning_rate * (values[index][len(values[index]) - 1] - prediction_for_row)
    return weights, threshold

def main():
    parser = argparse.ArgumentParser("Perceptron Program")
    parser.add_argument("--learning-rate", dest="learning_rate", help="The learning rate of the ML perceptron model", type=float)
    parser.add_argument("--num-passes", dest="num_passes", help="The number of passes for training", type=int)
    parser.add_argument("--threshold", dest="threshold", help="The linear separation threshold", type=float)
    args = parser.parse_args()

    learning_rate = args.learning_rate if args.learning_rate else 0.0002
    num_passes = args.num_passes if args.num_passes else 10000
    threshold = args.threshold if args.threshold else 0

    linearly_separated_data = [(1, 3, 0), (2, 2, 0), (3, 3, 0), (2, 4, 0), (1, 7, 0), (5, 3, 1), (6, 2, 1), (7, 1, 1), (8, 10, 1), (9, 0, 1)]

    values = linearly_separated_data

    weights, threshold = train(learning_rate, num_passes, threshold, values)
    print(weights)
    print(threshold)

    print(f'PREDICTION: {predictALabel(weights, [(-1, 5)], threshold)}')
if __name__ == "__main__":
    main()
