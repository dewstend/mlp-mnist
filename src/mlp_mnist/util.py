import matplotlib.pyplot as plt


def print_example(data):
    data = data.reshape(28, 28)  # Reshape to 28x28 grid

    # Plot the image
    plt.imshow(data, cmap="gray")
    plt.show()
