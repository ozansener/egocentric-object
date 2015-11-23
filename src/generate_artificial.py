from artificial_mnist import ArtificialMnist


data_generator = ArtificialMnist()
data_generator.get_n_random_samples(60000, './MNIST_a/train_data/')

data_generator.get_n_random_samples(10000, './MNIST_a/test_data/')

