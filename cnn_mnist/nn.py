import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix

tf.experimental.numpy.experimental_enable_numpy_behavior()

# learn images, learn lables, test ...
(li, ll), (ti, tl) = tf.keras.datasets.mnist.load_data()

print(tf.shape(li))

li = tf.reshape(li, (60000, 28, 28, 1)).astype('float32') / 255
ti = tf.reshape(ti, (10000, 28, 28, 1)).astype('float32') / 255

model = models.Sequential()

model.add(layers.Input(shape = (28, 28, 1)))

model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate = 0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_end = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        patience = 3,
        restore_best_weights = True
)

model.fit(li, ll, 
          epochs=20, 
          batch_size=64, 
          validation_split=0.2,
          callbacks = [early_end])

test_loss, test_acc = model.evaluate(ti, tl)

print(f"Test loss: {test_loss}")
print(f"Test acc: {test_acc}")

y_pred = model.predict(ti)
y_pred_classes = np.argmax(y_pred, axis=1)

if len(tl.shape) > 1 and tl.shape[1] > 1:
    y_true = np.argmax(tl, axis=1)
else:
    y_true = tl

print(classification_report(y_true, y_pred_classes))

precision = precision_score(y_true, y_pred_classes, average='macro')
sensitivity = recall_score(y_true, y_pred_classes, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

model.save("simple_mnist.keras")
