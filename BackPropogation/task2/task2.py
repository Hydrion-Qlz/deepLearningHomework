from utils import *
from model import *

train_images, train_labels = load_mnist_train()
test_images, test_labels = load_mnist_test()

# Initialize parameters
epochs = 30
batch_size = 32
learning_rate = 0.01
input_size = 28 * 28
hidden_size = 256
output_size = 10

# three_layer_model = ThreeLayerModel(input_size, hidden_size, output_size)
two_layer_model = TwoLayerModel(input_size, output_size)


def main():
    train_loss_lst = []
    test_loss_lst = []
    train_accuracy_lst = []
    test_accuracy_lst = []

    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        permutation = np.random.permutation(train_images.shape[0])
        shuffled_images = train_images[permutation]
        shuffled_labels = train_labels[permutation]

        for i in range(0, train_images.shape[0], batch_size):
            # Extract mini-batch
            X_batch = shuffled_images[i:i + batch_size]
            Y_batch = shuffled_labels[i:i + batch_size]

            # Forward and backward propagation on the mini-batch
            Z1, A1 = two_layer_model.forward(X_batch)
            dW1, db1 = two_layer_model.backward(X_batch, Y_batch, Z1, A1)

            # Update parameters
            two_layer_model.update_params(dW1, db1, learning_rate)

        train_loss_lst.append(compute_loss(train_images, train_labels, two_layer_model))
        test_loss_lst.append(compute_loss(test_images, test_labels, two_layer_model))

        train_accuracy_lst.append(compute_accuracy(train_images, train_labels, two_layer_model))
        test_accuracy_lst.append(compute_accuracy(test_images, test_labels, two_layer_model))
        print_epoch_result(epoch, test_accuracy_lst, test_loss_lst, train_accuracy_lst, train_loss_lst)

    plot_result_figure(train_loss_lst, test_loss_lst, train_accuracy_lst, test_accuracy_lst,
                       "Model Performance",
                       "./task2/result.png")


if __name__ == '__main__':
    main()
