import numpy as np
import theano
import theano.tensor as T
import theano.sparse
import sys
import time
import math


class RBM:
    """Restricted Boltzmann Machine applied to Collaborative Filtering.

    Attributes:
        weights
        bias_visible: bias of the layer containing visible units
        bias_hidden: bias of the layer containing hidden units
    """

    def __init__(self, n_items, n_labels, n_hidden=100):
        self.n_labels = n_labels
        n_visible = n_labels * n_items  # n_labels visible units per movie
        self.rng = T.shared_randomstreams.RandomStreams(seed=42)

        # Initializing weights and biases
        weights = np.array(np.random.normal(0, 0.1, size=(n_visible, n_hidden)), dtype=np.float32)
        h_bias = np.zeros(n_hidden, dtype=np.float32)
        v_bias = np.zeros(n_visible, dtype=np.float32)

        self.weights = theano.shared(value=weights, borrow=True)
        self.bias_visible = theano.shared(value=v_bias, borrow=True)
        self.bias_hidden = theano.shared(value=h_bias, borrow=True)

        # Initializing gradients
        weights_gradient = np.zeros(shape=(n_visible, n_hidden), dtype=np.float32)
        self.gradient_weights = theano.shared(value=weights_gradient, borrow=True)
        hidden_bias_gradient = np.zeros(n_hidden, dtype=np.float32)
        self.gradient_bias_hidden = theano.shared(value=hidden_bias_gradient, borrow=True)
        visible_bias_gradient = np.zeros(n_visible, dtype=np.float32)
        self.gradient_bias_visible = theano.shared(value=visible_bias_gradient, borrow=True)

    def train(self, train, items, test=None, momentum=0.5, learning_rate=0.0005, decay=0.0002, batch_size=10,
              n_epochs=100, verbose=True):
        """Trains the RBM from the `train` dataset.

        Arguments:
            train: dictionary where each key is a user id and each value is a list of tuples (movie_id, rating)
            items: list of movie id
            test: same format as train but for test data
                if not null, prints the test error at the end of each epoch
        """

        total_training_time = 0

        trainer = rbm.contrastive_divergence(learning_rate=learning_rate, decay=decay, momentum=momentum)
        predictor = rbm.get_predictor()

        movie_indices = {movie_id: i for i, movie_id in enumerate(items)}

        for epoch in range(n_epochs):
            print("Epoch %d" % epoch)
            current = time.time()

            # Train during an epoch
            for batch in self.make_batches(train.keys(), batch_size):
                user_vectors, masks = self.build_user_vectors(train, batch, batch_size, movie_indices)
                trainer(user_vectors, masks)

                sys.stdout.write('.')
                sys.stdout.flush()

            training_time = time.time() - current
            total_training_time += training_time
            print("\nTraining time: %F seconds" % training_time)

            # Predicting score on test

            if test:
                ratings = []
                predictions = []

                for batch in self.make_batches(test.keys(), batch_size):
                    user_vectors, _ = self.build_user_vectors(train, batch, batch_size, movie_indices)
                    user_predictions = self.expected_value(predictor(user_vectors))

                    indices = {user_id: i for i, user_id in enumerate(batch)}

                    for user_id in batch:
                        for movie_id, rating in test[user_id]:
                            user_prediction = user_predictions[indices[user_id]]
                            predicted = user_prediction[movie_indices[movie_id]]

                            ratings.append(float(rating))
                            predictions.append(predicted)

                ratings = np.array(ratings)
                predictions = np.array(predictions)

                mae = np.absolute(ratings - predictions).mean()
                rmse = math.sqrt(np.power(ratings - predictions, 2).mean())

                print("MAE: %f, RMSE: %f\n" % (mae, rmse))

        if verbose:
            print("Total training time for %d epochs: %F seconds" % (n_epochs, total_training_time))

    def get_predictor(self):
        """ Returns the function used for prediction. """
        visible_matrix = T.matrix()
        hidden, _ = self.sample_hidden(visible_matrix)
        visible_output, activations = self.sample_visible(hidden)
        return theano.function([visible_matrix], activations)

    def make_batches(self, data, batch_size):
        """ Returns batches from `data` with size `batch_size`. """
        batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        return batches

    def build_user_vectors(self, train, batch, batch_size, movie_indices):
        """
            Returns:
                user_vectors: a list of vectors representing the visible layer for each user in `batch`
                masks: the corresponding masks where values equal 1 if the user rated the corresponding movie,
                    0 otherwise
        """
        size = min(len(batch), batch_size)
        user_vectors = {}
        masks = {}

        for user_id in batch:
            user_ratings = [0.] * len(items)
            mask = [0] * (len(items) * self.n_labels)

            for movie_id, rating in train[user_id]:
                user_ratings[movie_indices[movie_id]] = float(rating)

                for i in range(self.n_labels):
                    mask[self.n_labels * movie_indices[movie_id] + i] = 1

            user_vector = [0] * (len(user_ratings) * self.n_labels)
            ratings = list(np.linspace(0.5, 5, 10))
            rating_index = {rating: i for i, rating in enumerate(ratings)}

            for i, rating in enumerate(user_ratings):
                if rating != 0.:
                    user_vector[(i * self.n_labels) + rating_index[rating]] = 1

            user_vector = np.array(user_vector).reshape(1, -1).astype('float32')

            user_vectors[user_id] = user_vector
            masks[user_id] = mask

        user_vectors = np.array([user_vectors[id] for id in batch]).reshape(size, len(items)*self.n_labels)
        masks = np.array([masks[id] for id in batch]).reshape(size, len(items)*self.n_labels).astype('float32')

        return user_vectors, masks

    def contrastive_divergence(self, learning_rate=0.000025, decay=0., momentum=0.):
        """ Performs the contrastive divergence to update the weights and biases of the RBM. """
        user_vectors = T.matrix()
        masks = T.matrix()

        # Contrastive divergence
        visible_1 = user_vectors
        hidden_1, _ = self.sample_hidden(visible_1)
        visible_2, visible_2_activations = self.sample_visible(hidden_1)
        hidden_2, hidden_2_activations = self.sample_hidden(visible_2)

        # Update weights and biases
        (gradient_weights_new, gradient_bias_visible_new, gradient_bias_hidden_new) = \
            self.gradient(visible_1, hidden_1, visible_2, hidden_2_activations, masks)

        gradient_weights_new -= decay * self.weights

        updates = [
            (self.weights, T.cast(self.weights + (momentum * self.gradient_weights) +
                                  (gradient_weights_new * learning_rate), 'float32')),
            (self.bias_visible, T.cast(self.bias_visible + (momentum * self.gradient_bias_visible) +
                                       (gradient_bias_visible_new * learning_rate), 'float32')),
            (self.bias_hidden, T.cast(self.bias_hidden + (momentum * self.gradient_bias_hidden) +
                                      (gradient_bias_hidden_new * learning_rate), 'float32')),
            (self.gradient_weights, T.cast(gradient_weights_new, 'float32')),
            (self.gradient_bias_hidden, T.cast(gradient_bias_hidden_new, 'float32')),
            (self.gradient_bias_visible, T.cast(gradient_bias_visible_new, 'float32'))
        ]

        return theano.function([user_vectors, masks], updates=updates)

    def sample_hidden(self, visible):
        """ Samples the hidden layer. """
        activations = T.nnet.sigmoid(T.dot(visible, self.weights) + self.bias_hidden)
        hidden_sample = self.rng.binomial(size=activations.shape, p=activations, dtype=theano.config.floatX)
        return hidden_sample, activations

    def sample_visible(self, hidden):
        """ Samples the visible layer. """
        activations = T.nnet.sigmoid(T.dot(hidden, self.weights.T) + self.bias_visible)
        normalization_factor = activations.reshape((-1, self.n_labels)).sum(axis=1).reshape((-1, 1)) \
            * T.ones(self.n_labels)
        activations = activations / normalization_factor.reshape(activations.shape)

        visible_sample = self.rng.binomial(size=activations.shape, p=activations, dtype=theano.config.floatX)

        return visible_sample, activations

    def expected_value(self, prediction):
        """ Returns the expected value for each movie for each user. """
        labels = list(np.linspace(0.5, 5, 10))
        users = np.array((prediction.reshape(-1, self.n_labels) * labels).sum(axis=1))

        ret = np.array(users).reshape(prediction.shape[0], prediction.shape[1] / self.n_labels)
        return ret

    def gradient(self, visible_1, hidden_1, visible_2, visible_2_activations, masks):
        """ Computes the gradient for the weights and biases. """
        def outer(x, y):
            return x[:, :, np.newaxis] * y[:, np.newaxis, :]

        gradient_mask = outer(masks, hidden_1)

        gradient_weights = ((outer(visible_1, hidden_1) - outer(visible_2, visible_2_activations))
                            * gradient_mask).mean(axis=0)
        gradient_bias_visible = ((visible_1 * masks) - (visible_2 * masks)).mean(axis=0)
        gradient_bias_hidden = (hidden_1 - visible_2_activations).mean(axis=0)

        return gradient_weights, gradient_bias_visible, gradient_bias_hidden

if __name__ == "__main__":
    n_labels = 10  # all possible ratings: 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5
    output_file = "storage.npz"

    # Load the datasets

    npz_file = np.load(output_file)

    items = list(npz_file['all_movies'])
    train = npz_file['profiles'].item()
    test = npz_file['tests'].item()

    print("Number of users in train: %d" % len(train.keys()))
    print("Number of users in test: %d" % len(test.keys()))
    print("Number of movies: %d" % len(items))

    # Computing train and test size

    train_size = 0
    test_size = 0

    for i in train.keys():
        train_size += len(train[i])

    for i in test.keys():
        test_size += len(test[i])

    print("Number of ratings in train: %d" % train_size)
    print("Number of ratings in test: %d\n" % test_size)

    # Train the model

    rbm = RBM(len(items), n_labels, n_hidden=100)
    rbm.train(train, items, test=test, momentum=0.5, learning_rate=0.0005, decay=0.0002, batch_size=10, n_epochs=100)
