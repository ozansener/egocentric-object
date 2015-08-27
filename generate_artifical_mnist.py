from artificial_mnist import ArtificialMnist

# First generate images

data_generator = ArtificialMnist()
data_generator.get_n_random_samples(1000,
                                    './data/artificial_mnist/train_data/')

data_generator.get_n_random_samples(100,
                                    './data/artificial_mnist/test_data/')

# Then convert it to torch datatype