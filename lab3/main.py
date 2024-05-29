import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Main.Task №1 and №2
def stochastic_gradient_descent(
        x, y,
        start,
        batch=2,
        eps=1e-5,
        epochs=50,
        learning_rate_method='standard'
):
    it = 0
    w, b = start

    learning_rate = {
        'standard': lambda *args: 1e-2,
        'step_func': lambda it: max(1e-3, 0.1 - it * 1e-3)
    }

    for epoch in range(epochs):
        for i in range(0, x.shape[0], batch):
            it += 1

            xi = x[i:i + batch]
            yi = y[i:i + batch]

            h = learning_rate[learning_rate_method](it)

            predict = np.dot(xi, w) + b

            w_k = w - h * (2 / batch) * np.dot(xi.T, predict - yi)
            b_k = b - h * (2 / batch) * np.sum(predict - yi)

            if (np.linalg.norm(w_k - w) < eps
                    or np.linalg.norm(b_k - b) < eps):
                return {'w': w, 'b': b, 'it': it}

            w, b = w_k, b_k

    return {'w': w, 'b': b, 'it': it}


# Main.Task №3
def train_with_tensorflow_optimizer(X, Y, optimizer, epochs=50, batch_size=32):
    X = tf.constant(X, dtype=tf.float32)
    Y = tf.constant(Y, dtype=tf.float32)

    w = tf.Variable(tf.random.normal([1, 1]), name='weight')
    b = tf.Variable(tf.random.normal([1]), name='bias')

    def model(x):
        return tf.matmul(x, w) + b

    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    for epoch in range(epochs):
        for i in range(0, X.shape[0], batch_size):
            xi = X[i:i + batch_size]
            yi = Y[i:i + batch_size]
            with tf.GradientTape() as tape:
                y_pred = model(xi)
                loss = loss_fn(yi, y_pred)
            gradients = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(zip(gradients, [w, b]))

    return w.numpy(), b.numpy()


def create_plot(x, y, w, b):
    fig, ax = plt.subplots()
    for point in zip(x, y):
        ax.scatter(point[0], point[1])
    ax.plot(x, w * x + b)


def main():
    """
    :MAIN:
    Stochastic gradient descent:
    np.random.seed(234)
    X = 2 * np.random.rand(200, 1)
    Y = 6 * X + np.random.randn(200, 1) + 12
    """

    np.random.seed(234)
    X = 2 * np.random.rand(200, 1)
    Y = 6 * X + np.random.randn(200, 1) + 12

    start = (0, 0)
    for batch in [1, 32, 50, 64, 100]:
        result = stochastic_gradient_descent(
            X, Y, start, epochs=10000, batch=batch, learning_rate_method='step_func')

        print(result)

        create_plot(X, Y, result.get('w'), result.get('b'))

    print("========== TensorFlow optimizers ==========")

    optimizers = {
        'SGD': tf.optimizers.SGD(learning_rate=0.01),
        'Momentum': tf.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'Nesterov': tf.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        'AdaGrad': tf.optimizers.Adagrad(learning_rate=2.0),
        'RMSProp': tf.optimizers.RMSprop(learning_rate=0.01),
        'Adam': tf.optimizers.Adam(learning_rate=0.01)
    }

    for name, optimizer in optimizers.items():
        w, b = train_with_tensorflow_optimizer(X, Y, optimizer, epochs=1000, batch_size=32)
        print(f"{name} - Weight: {w.flatten()}, Bias: {b}")
        create_plot(X, Y, w, b)

    plt.show()


if __name__ == '__main__':
    main()
