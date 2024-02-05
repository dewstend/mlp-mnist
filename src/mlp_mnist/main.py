from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_data():
    mnist = fetch_openml("mnist_784")
    X = mnist.data
    y = mnist.target
    X = X.astype("float32")
    y = y.astype("int32")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    return X, y


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, random_state=42
    )
    print("first 100 training features:", X_train[0:100])
    print("first 100 training labels:", y_train[0:100])


if __name__ == "__main__":
    main()
