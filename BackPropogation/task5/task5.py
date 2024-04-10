from model import *
from utils import *

train_images, train_labels = load_mnist_train()
test_images, test_labels = load_mnist_test()

# Initialize parameters
epochs = 10
batch_size = 32
learning_rate = 0.01
input_size = 28 * 28
hidden_size = 256
output_size = 10


def train_and_test(model, figure_title, save_path, batch_size=batch_size, epochs=epochs):
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
            Z1, A1, Z2, A2 = model.forward(X_batch)
            dW1, db1, dW2, db2 = model.backward(X_batch, Y_batch, Z1, A1, Z2, A2)

            # Update parameters
            model.update_params(dW1, db1, dW2, db2, learning_rate)

        train_loss_lst.append(compute_loss(train_images, train_labels, model))
        test_loss_lst.append(compute_loss(test_images, test_labels, model))

        train_accuracy_lst.append(compute_accuracy(train_images, train_labels, model))
        test_accuracy_lst.append(compute_accuracy(test_images, test_labels, model))
        print_epoch_result(epoch, test_accuracy_lst, test_loss_lst, train_accuracy_lst, train_loss_lst)

    plot_result_figure(train_loss_lst, test_loss_lst, train_accuracy_lst, test_accuracy_lst,
                       figure_title,
                       save_path)


if __name__ == '__main__':
    # Constant Learning rate
    print("Training Model Using Constant Learning Rate")
    constant_model = ThreeLayerModel(input_size, hidden_size, output_size)
    train_and_test(model=constant_model,
                   figure_title="Model Performance Using Constant Learning Rate",
                   save_path="./task5/constant-learning-rate.png")

    # Momentum
    print("Training Model Using Momentum")
    momentum_model = ThreeLayerModel_Momentum(input_size, hidden_size, output_size)
    train_and_test(model=momentum_model,
                   figure_title="Model Performance Using Momentum",
                   save_path="./task5/momentum.png")

    # RMSProp
    print("Training Model Using RMSProp")
    RMSProp_model = ThreeLayerModel_RMSProp(input_size, hidden_size, output_size)
    train_and_test(model=RMSProp_model,
                   figure_title="Model Performance Using RMSProp",
                   save_path="./task5/RMSProp.png")

    # Adam
    print("Training Model Using Adam")
    Adam_model = ThreeLayerModel_Adam(input_size, hidden_size, output_size)
    train_and_test(model=Adam_model,
                   figure_title="Model Performance Using RMSProp",
                   save_path="./task5/Adam.png")
