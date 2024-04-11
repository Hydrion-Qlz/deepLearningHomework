from model import *
from utils import *

train_images, train_labels = load_mnist_train()
test_images, test_labels = load_mnist_test()

# Initialize parameters
epochs = 10
batch_size = 32
learning_rate = 0.01
input_size = 28 * 28
hidden_size1 = 128
hidden_size2 = 32
output_size = 10

# three_layer_model = ThreeLayerModel(input_size, hidden_size, output_size)
two_layer_model = TwoLayerModel(input_size, output_size)
four_layer_model = FourLayerModel(input_size, hidden_size1, hidden_size2, output_size)


def train_and_test():
    train_loss_lst_2 = []
    test_loss_lst_2 = []
    train_accuracy_lst_2 = []
    test_accuracy_lst_2 = []

    train_loss_lst_4 = []
    test_loss_lst_4 = []
    train_accuracy_lst_4 = []
    test_accuracy_lst_4 = []

    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        permutation = np.random.permutation(train_images.shape[0])
        shuffled_images = train_images[permutation]
        shuffled_labels = train_labels[permutation]

        for i in range(0, train_images.shape[0], batch_size):
            # Extract mini-batch
            X_batch = shuffled_images[i:i + batch_size]
            Y_batch = shuffled_labels[i:i + batch_size]

            # two layer model
            Z1, A1 = two_layer_model.forward(X_batch)
            dW1, db1 = two_layer_model.backward(X_batch, Y_batch, Z1, A1)

            two_layer_model.update_params(dW1, db1, learning_rate)

            # four layer model
            Z1, A1, Z2, A2, Z3, A3 = four_layer_model.forward(X_batch)
            dW1, db1, dW2, db2, dW3, db3 = four_layer_model.backward(X_batch, Y_batch, Z1, A1, Z2, A2, Z3, A3)

            four_layer_model.update_params(dW1, db1, dW2, db2, dW3, db3, learning_rate)

        # two layer model
        train_loss_lst_2.append(compute_loss(train_images, train_labels, two_layer_model))
        test_loss_lst_2.append(compute_loss(test_images, test_labels, two_layer_model))

        train_accuracy_lst_2.append(compute_accuracy(train_images, train_labels, two_layer_model))
        test_accuracy_lst_2.append(compute_accuracy(test_images, test_labels, two_layer_model))
        print_epoch_result(epoch, test_accuracy_lst_2, test_loss_lst_2, train_accuracy_lst_2, train_loss_lst_2)

        # four layer model
        train_loss_lst_4.append(compute_loss(train_images, train_labels, four_layer_model))
        test_loss_lst_4.append(compute_loss(test_images, test_labels, four_layer_model))

        train_accuracy_lst_4.append(compute_accuracy(train_images, train_labels, four_layer_model))
        test_accuracy_lst_4.append(compute_accuracy(test_images, test_labels, four_layer_model))
        print_epoch_result(epoch, test_accuracy_lst_4, test_loss_lst_4, train_accuracy_lst_4, train_loss_lst_4)

    plot_result_figure(train_loss_lst_2, test_loss_lst_2, train_accuracy_lst_2, test_accuracy_lst_2,
                       "Two Layer Model Performance",
                       "./task2/twoLayerModel.png")
    plot_result_figure(train_loss_lst_4, test_loss_lst_4, train_accuracy_lst_4, test_accuracy_lst_4,
                       "Four Layer Model Performance",
                       "./task2/fourLayerModel.png")


if __name__ == '__main__':
    train_and_test()
