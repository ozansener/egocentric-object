local parser  = require('src.dataset_parser')


parser.read_and_save("data/artificial_mnist/train_data/",
    "data/artificial_mnist_train.t7")


parser.read_and_save("data/artificial_mnist/test_data/",
    "data/artificial_mnist_test.t7")
