import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split



# Load the dataset
df = pd.read_csv('spam2.csv', encoding='ISO-8859-1')

# Extract features and labels
X = df['v2']  # Assuming 'v2' is the column with the email text
y = df['v1']  # Assuming 'v1' is the column with labels

# Split the data into training, validation, and test sets
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.40, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_remaining, y_remaining, test_size=0.50, random_state=42)

# Create TensorFlow datasets
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
validation_data = tf.data.Dataset.from_tensor_slices((X_validation, y_validation))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Define batch size
batch_size = 512

# Shuffle and batch the training data
train_data = train_data.shuffle(buffer_size=10000).batch(batch_size)

# Batch the validation and test data
validation_data = validation_data.batch(batch_size)
test_data = test_data.batch(batch_size)

# TensorFlow Hub layer
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)


# Define the model with an LSTM layer and Dropout
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Reshape(target_shape=(50, 1)),  # Reshape for LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data,
                    epochs=10,
                    validation_data=validation_data,
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data, verbose=2)

# Print results
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

# Save as HDF5
model.save('my_model.h5')