import ydf
import pandas as pd
import numpy as np
import tensorflow as tf

tf.experimental.numpy.experimental_enable_numpy_behavior()

# learn images, learn lables, test ...
(li, ll), (ti, tl) = tf.keras.datasets.mnist.load_data()

li_flat = li.reshape(60000, 28*28).astype('float32') / 255
ti_flat = ti.reshape(10000, 28*28).astype('float32') / 255

feature_names = [f'pixel_{i}' for i in range(28*28)]

train_df = pd.DataFrame(li_flat, columns=feature_names)
train_df['label'] = ll.astype(int)

test_df = pd.DataFrame(ti_flat, columns=feature_names)
test_df['label'] = tl.astype(int)

model = ydf.RandomForestLearner(
    label="label",
    task=ydf.Task.CLASSIFICATION,
    num_trees=100,
    max_depth=20,
    winner_take_all=False
).train(train_df)

predictions_proba = model.predict(test_df)
pred_classes = np.argmax(predictions_proba, axis=1)
accuracy = np.mean(pred_classes == test_df['label'].values)
print(f"Test accuracy: {accuracy:.4f}")

model_path = "mnist_random_forest_ydf"
model.save(model_path)
