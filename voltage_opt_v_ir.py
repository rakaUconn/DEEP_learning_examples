## we want to use automatic differentiation to learn the parameters (R) of this simple model,
# we can set up an optimization problem where we try to minimize the difference between the actual
# and predicted values of  of V given  I and R

import tensorflow as tf
# define input current and voltage
current = tf.constant([1.0,2.0,3.0,6.0,8.0])
voltage = tf.constant([2.0,3.0,5.9,12.5,16.1])


# Initialize resistance (the parameter to be learned)
resistance = tf.Variable(1.0, dtype=tf.float32)
# define function that predict voltage
def predict_voltage(current,resistance):
    return current*resistance

# Define a loss function (mean squared error)
def loss(predict_voltage,actual_voltage):
    return tf.reduce_mean(tf.square(predict_voltage-actual_voltage))

# Use gradient descent to optimize the resistance value
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    with tf.GradientTape() as Tape:
        predict_voltages=predict_voltage(current,resistance)
        current_loss=loss(predict_voltages,voltage)
    gradients=Tape.gradient(current_loss,[resistance])
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    optimizer.apply_gradients(zip(gradients,[resistance]))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss: {current_loss.numpy()}, Learned Resistance: {resistance.numpy()}")

print(f"Final learned resistance: {resistance.numpy()}")
