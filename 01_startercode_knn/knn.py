import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = np.array(features)
        self.labels = np.array(labels)

    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds the k nearest neighbours in the training set.
        It needs to return a list of labels of these k neighbours. When there is a tie in distance, 
        prioritize examples with a smaller index.
        :param point: List[float]
        :return:  List[int]
        """
        # Create a vector of length similar to the features vector
        np_points = np.repeat([np.array(point)], self.features.shape[0], axis=0)

        # Calculate distance
        distances = self.distance_function(self.features, np_points)

        # Sort labels w.r.t distances and return the first k labels
        return self.labels[distances.argsort()[:self.k]]

    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predicted label for that testing data point (you can assume that k is always a odd number).
        Thus, you will get N predicted label for N test data point.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        # finds the majority label
        majority_label = lambda k_neighbors: Counter(k_neighbors).most_common()[0][0]

        # get k neighbors for all data points
        k_neighbors = np.apply_along_axis(self.get_k_neighbors, 1, features)

        # creates a list predicted labels
        return np.apply_along_axis(majority_label, 1, k_neighbors)


if __name__ == '__main__':
    print(np.__version__)
