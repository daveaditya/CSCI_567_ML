import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n)  # this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
    # according to some distribution, first generate a random number between 0 and
    # 1 using generator.rand(), then find the the smallest index n so that the
    # cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    center_points = list()
    centers = list()

    center_points.append(x[p])
    centers.append(p)

    # there will be k clusters
    for _ in range(1, n_cluster):

        distance_to_closest_centroid = list()

        # for every point find the closest distance center points
        for i in range(len(x)):

            # find the minimum distanced center point
            minimum_distance = float("inf")
            for center_point in center_points:
                distance = np.sum((x[i] - center_point) ** 2)
                if distance < minimum_distance:
                    minimum_distance = distance

            # Add to closest centroid distance
            distance_to_closest_centroid.append(minimum_distance)

        # Pre-calculate sum of distances
        sum_of_distances = sum(distance_to_closest_centroid)

        # Update distance to closest centroit
        for m in range(len(distance_to_closest_centroid)):
            distance_to_closest_centroid[m] = distance_to_closest_centroid[m] / \
                sum_of_distances

        # randomly select a center point
        index = None
        r = generator.rand()
        cumulative_prob = 0
        for i in range(len(distance_to_closest_centroid)):
            cumulative_prob += distance_to_closest_centroid[i]
            if cumulative_prob > r:
                index = i
                break

        centers.append(index)
        center_points.append(x[index])

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################

        # Initialize centroids
        centroids = np.zeros([self.n_cluster, D])
        for i in range(self.n_cluster):
            centroids[i] = x[self.centers[i]]

        y = np.zeros(N)

        # Calculate initial distortion
        distortion = np.sum([np.sum(np.power((x[y == i] - centroids[i]), 2))
                            for i in range(self.n_cluster)]) / N

        temp_iterations = None
        for z in range(self.max_iter):
            temp_iterations = z

            # Find closest centers
            y = np.argmin(
                np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2), axis=0)

            # Calculate updated distortion
            distortion_updated = np.sum(
                [np.sum(np.power((x[y == i] - centroids[i]), 2)) for i in range(self.n_cluster)]) / N

            # If the average K means objective changes less than e than stop
            if np.absolute(distortion - distortion_updated) <= self.e:
                break

            # Update distortion
            distortion = distortion_updated

            # Calculate and set new centroids
            new_centroids = np.array(
                [np.mean(x[y == i], axis=0) for i in range(self.n_cluster)])
            index = np.where(np.isnan(new_centroids))
            new_centroids[index] = centroids[index]
            centroids = new_centroids

        # Set max iterations to the one that actually took
        self.max_iter = temp_iterations

        return centroids, y, self.max_iter


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################

        k_means = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)

        centroids, assignment, max_iterations = k_means.fit(x, centroid_func)

        # Assign labels
        assigned_label = [[] for i in range(self.n_cluster)]
        for i in range(N):
            assigned_label[assignment[i]].append(y[i])

        centroid_labels = np.zeros([self.n_cluster])
        for i in range(self.n_cluster):
            centroid_labels[i] = np.argmax(np.bincount(assigned_label[i]))

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################

        # Find L2 Norm
        l2_norm = np.zeros([self.n_cluster, N])
        for k in range(self.n_cluster):
            l2_norm[k] = np.sqrt(np.sum(np.power((x - self.centroids[k]), 2), axis=1))

        # Get the nearest centroid
        nearest_centroid = np.argmin(l2_norm, axis=0)

        # Assign labels
        resultant_labels = [[] for i in range(N)]
        for index in range(N):
            resultant_labels[index] = self.centroid_labels[nearest_centroid[index]]

        return np.array(resultant_labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################

    R, G, B = image.shape
    reshaped_image = image.reshape(R * G, B)
    nearest_index = np.argmin(np.sum((np.power(
        reshaped_image - np.expand_dims(code_vectors, axis=1), 2)), axis=2), axis=0)
    transformed_image = code_vectors[nearest_index].reshape(R, G, B)
    return transformed_image
