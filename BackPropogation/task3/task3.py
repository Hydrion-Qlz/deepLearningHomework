from model import *

train_images, train_labels = load_mnist_train()
test_images, test_labels = load_mnist_test()

# Initialize parameters
# epochs = 10
# batch_size = 32
learning_rate = 0.01
input_size = 28 * 28
hidden_size = 256
output_size = 10


def train_and_test(model, batch_size, epochs, figure_title, save_path):
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
    # Mini-batch Gradient - set batch size to 32
    print("Training Model Using Mini-batch Gradient")
    Mini_batch_gradient_model = ThreeLayerModel(input_size, hidden_size, output_size)
    train_and_test(model=Mini_batch_gradient_model,
                   batch_size=32,
                   epochs=10,
                   figure_title="Model Performance Using Mini-batch Gradient",
                   save_path="./task3/Mini-batch-Gradient.png")

    # # BGD - set batch size to dataset size
    # print("Training Model Using BGD")
    # BGD_model = ThreeLayerModel(input_size, hidden_size, output_size)
    # train_and_test(model=BGD_model,
    #                batch_size=train_images.shape[0],
    #                epochs=10,
    #                figure_title="Model Performance Using BGD",
    #                save_path="./task3/BGD.png")
    #
    # # SGD - set batch size to 1
    # print("Training Model Using SGD")
    # SGD_model = ThreeLayerModel(input_size, hidden_size, output_size)
    # train_and_test(model=SGD_model,
    #                batch_size=1,
    #                epochs=10,
    #                figure_title="Model Performance Using SGD",
    #                save_path="./task3/SGD.png")
