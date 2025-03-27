import tensorflow as tf
from tensorflow.keras import layers, models

tf.experimental.numpy.experimental_enable_numpy_behavior()

# learn images, learn lables, test ...
(li, ll), (ti, tl) = tf.keras.datasets.mnist.load_data()

print(tf.shape(li))

li = tf.reshape(li, (60000, 28, 28, 1)).astype('float32') / 255
ti = tf.reshape(ti, (10000, 28, 28, 1)).astype('float32') / 255

# Create SimpleNet v1 with BN layers
model = models.Sequential()

model.add(layers.Input(shape = (28, 28, 1)))

# Block 1
model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 3
model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# 1x1
model.add(layers.Conv2D(256, kernel_size=(1, 1)))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(256, kernel_size=(1, 1)))
model.add(layers.BatchNormalization())  # BN layer
model.add(layers.Activation('relu'))

# Final 1x1 Conv for classification
model.add(layers.Conv2D(10, kernel_size=(1, 1)))  # 10 classes for MNIST

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Activation('softmax'))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 25 == 0:
        return lr * 0.1
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

early_end = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        patience = 5,
        restore_best_weights = True
)

# Create data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Manually split the data into training and validation sets
validation_split = 0.1
val_samples = int(validation_split * li.shape[0])
train_samples = li.shape[0] - val_samples

# Shuffle indices
indices = tf.random.shuffle(tf.range(li.shape[0]))
train_indices = indices[:train_samples]
val_indices = indices[train_samples:]

# Split images and labels
train_images = tf.gather(li, train_indices)
train_labels = tf.gather(ll, train_indices)
val_images = tf.gather(li, val_indices)
val_labels = tf.gather(ll, val_indices)

# Create training dataset with augmentation
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# Create validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Train the model with both callbacks and proper datasets
model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[early_end, lr_callback]
)

test_loss, test_acc = model.evaluate(ti, tl)

print(f"Test loss: {test_loss}")
print(f"Test acc: {test_acc}")

model.save("simplenet_mnist.keras")

# Print model summary to verify architecture
model.summary()
