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

# Block 1
model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))

model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 3
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.Dropout(rate=0.05))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(128, kernel_size=(1, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(128, kernel_size=(1, 1)))
model.add(layers.Dropout(rate=0.05))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(10, kernel_size=(1, 1)))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Activation('softmax'))

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.0195, momentum = 0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_end = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        patience = 3,
        restore_best_weights = True
)

model.fit(li, ll, 
          epochs=100, 
          batch_size=64, 
          validation_split=0.1,
          callbacks = [early_end])

test_loss, test_acc = model.evaluate(ti, tl)

print(f"Test loss: {test_loss}")
print(f"Test acc: {test_acc}")

model.save("simple_mnist.keras")
