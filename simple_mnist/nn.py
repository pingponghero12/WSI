import tensorflow as tf
from tensorflow.keras import layers, models

tf.experimental.numpy.experimental_enable_numpy_behavior()

# learn images, learn lables, test ...
(li, ll), (ti, tl) = tf.keras.datasets.mnist.load_data()

print(tf.shape(li))

li = tf.reshape(li, (60000, 28, 28, 1)).astype('float32') / 255
ti = tf.reshape(ti, (10000, 28, 28, 1)).astype('float32') / 255

model = models.Sequential()

model.add(layers.Input(shape = (28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_end = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        patience = 3,
        restore_best_weights = True
)

model.fit(li, ll, 
          epochs=10, 
          batch_size=64, 
          validation_split=0.2,
          callbacks = [early_end])

test_loss, test_acc = model.evaluate(ti, tl)

print(f"Test loss: {test_loss}")
print(f"Test acc: {test_acc}")

model.save("simple_mnist.keras")
