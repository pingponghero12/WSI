import tensorflow as tf
import pandas as pd
import tensorflow_decision_forests as tfdf
from tensorflow.keras import layers, models

tf.experimental.numpy.experimental_enable_numpy_behavior()

# learn images, learn lables, test ...
(li, ll), (ti, tl) = tf.keras.datasets.mnist.load_data()

print(tf.shape(li))

li = tf.reshape(li, (60000, 28, 28, 1)).astype('float32') / 255
ti = tf.reshape(ti, (10000, 28, 28, 1)).astype('float32') / 255

li_flat = li.reshape(60000, 28*28).astype('float32') / 255
ti_flat = ti.reshape(10000, 28*28).astype('float32') / 255

feature_names = [f'pixel_{i}' for i in range(28*28)]

train_df = pd.DataFrame(li_flat, columns=feature_names)
train_df['label'] = ll

test_df = pd.DataFrame(ti_flat, columns=feature_names)
test_df['label'] = tl

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='label')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='label')

model = tfdf.keras.RandomForestModel(
        verbose = 2,
        task = tfdf.keras.Task.CLASSIFICATION,
        num_trees = 100,
        max_depth = 20)

model.fit(train_ds)

evaluation = model.evaluate(test_ds, return_dict=True)

y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

acc = accuracy_score(tl, y_pred_classes)
print(f"Test accuracy (from predictions): {acc}")

model.save("mnist_random_forest.keras")
print("Model saved to 'mnist_random_forest'")

print(model.summary())
