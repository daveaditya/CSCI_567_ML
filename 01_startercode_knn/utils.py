import numpy as np
from numpy.ma.core import logical_and
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################


def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)

    # Convert to numpy array
    arr_real_labels = np.array(real_labels)
    arr_predicted_labels = np.array(predicted_labels)

    # Get true positive, false positive and false negative
    true_positive = np.sum(np.logical_and(
        arr_predicted_labels == 1, arr_real_labels == 1))
    false_positive = np.sum(np.logical_and(
        arr_predicted_labels == 1, arr_real_labels == 0))
    false_negative = np.sum(np.logical_and(
        arr_predicted_labels == 0, arr_real_labels == 1))

    # Calculate F1 score and return
    return true_positive / (true_positive + ((false_positive + false_negative) / 2))


class Distances:
    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.cbrt(np.sum(np.power(np.array(point1) - np.array(point2), 3), axis=1))

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.sqrt(np.sum(np.power(np.array(point1) - np.array(point2), 2), axis=1))

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        arr_point1 = np.array(point1)
        arr_point2 = np.array(point2)
        ed_point1 = np.sqrt(np.sum(np.power(arr_point1, 2)))
        ed_point2 = np.sqrt(np.sum(np.power(arr_point2, 2)))

        if ed_point1 == 0 or ed_point2 == 0:
            return 1
        else:
            return 1 - ((np.einsum('ij,ij->i', arr_point1, arr_point2))/(ed_point1 * ed_point2))


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
                (this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """

        # create an empty numpy array
        results = np.empty((0, 4))

        for distance_func_name in distance_funcs.keys():

            for k in range(1, 30):
                # create an instance of knn
                knn = KNN(k, distance_funcs[distance_func_name])

                # train the knn classifier
                knn.train(x_train, y_train)

                # make predictions
                predictions = knn.predict(x_val)

                # get F1 score
                score = f1_score(y_val, predictions)

                results = np.append(results, [[distance_func_name,
                                    score,
                                    k,
                                    knn]], axis=0)

        # sort the result based on f1 score in non-increasing order
        results = results[np.argsort(results[:,1], kind='stable')[::-1]]

        # sort the results based on distance_function
        distance_funcs_priority = {'euclidean': 0,
                                   'Minkowski': 1, 'cosine_dist': 2}
        results = results[np.argsort(
            np.array(distance_funcs_priority[df[0]] for df in results), kind='stable')]

        # sort the result based on k neighbors in non-decreasing order
        results = results[np.argsort(results[:,3], kind='stable')]

        # the first model is the one with best parameters
        best = results[0]

        # You need to assign the final values to these variables
        self.best_k = best[2]
        self.best_distance_function = best[0]
        self.best_model = best[3]

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """

        # create an empty numpy array
        results = np.empty((0, 5))

        for scaling_class in scaling_classes:

            # Select the scaler based on name
            scaler = None
            if scaling_class == 'min_max_scale':
                scaler = MinMaxScaler()
            else:
                scaler = NormalizationScaler()

            # scale the train and validation sets
            scaled_x_train = scaler(x_train)
            scaled_x_val = scaler(x_val)

            for distance_func_name in distance_funcs.keys():

                for k in range(1, 30):
                    # create an instance of knn
                    knn = KNN(k, distance_funcs[distance_func_name])

                    # train the knn classifier
                    knn.train(scaled_x_train, y_train)

                    # make predictions
                    predictions = knn.predict(scaled_x_val)

                    # get F1 score
                    score = f1_score(y_val, predictions)

                    results = np.append(results, [[
                        scaling_class,
                        distance_func_name,
                        k,
                        score,
                        knn
                    ]], axis=0)

        # sort the result based on f1 score in non-increasing order
        results = results[np.argsort(results[:,1], kind='stable')[::-1]]

        # sort the results based on scaling class
        scaling_class_priority = {'min_max_scale': 0, 'normalize': 1}
        results = results[np.argsort(
            np.array(scaling_class_priority[df[0]] for df in results), kind='stable')]

        # sort the results based on distance_function
        distance_funcs_priority = {'euclidean': 0,
                                   'Minkowski': 1, 'cosine_dist': 2}
        results = results[np.argsort(
            np.array(distance_funcs_priority[df[0]] for df in results), kind='stable')]

        # sort the result based on k neighbors in non-decreasing order
        results = results[np.argsort(results[:,3], kind='stable')]

        # the first model is the one with best parameters
        best = results[0]

        # You need to assign the final values to these variables
        self.best_k = best[2]
        self.best_distance_function = best[1]
        self.best_scaler = best[0]
        self.best_model = best[4]


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        # convert features to numpy array
        arr_features = np.array(features)

        # normalize function that will work on each data point
        def scale(features):
            """Scales features based on normalize logic

            Args:
                features ([List[float]]): Each feature from the features

            Returns:
                [List[float]]: Normalized feature
            """
            return features if np.all(features == 0) else features / np.sqrt(np.sum(np.power(features, 2)))

        # scale features and return
        return np.apply_along_axis(scale, 1, arr_features)


class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
            For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
            For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
            the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
            This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
            The minimum value of this feature is thus min=-1, while the maximum value is max=2.
            So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
            leading to 1, 0, and 0.333333.
            If max happens to be same as min, set all new values to be zero for this feature.
            (For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        # convert features to numpy array
        arr_features = np.array(features)

        def scale(feature):
            """Scales the feature based on min-max scaler logic

            Args:
                feature ([List[float]]): Single feature

            Returns:
                [List[float]]: Scaled feature
            """
            min = np.min(feature)
            max = np.max(feature)
            diff = max - min

            def scaler(feature): return 0 if min == max else (
                feature - min) / (diff)
            return np.apply_along_axis(scaler, 0, feature)

        # apply scaler to all features and return the final features
        return np.apply_along_axis(scale, 1, arr_features)
