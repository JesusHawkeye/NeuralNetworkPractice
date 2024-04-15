import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


"""
Sources for this code:
https://stackoverflow.com/questions/66714485/how-can-i-train-my-cnn-model-on-dataset-from-a-csv-file


"""


# Set tensorflow to run on my GPU (M2 Max on Apple Silicon)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)




# Load the dataset from CSV
csv_path = '../data/GTZAN/features_3_sec.csv'
dataframe = pd.read_csv(csv_path)

# Encode the categorical 'label' column
label_encoder = LabelEncoder()
dataframe['label'] = label_encoder.fit_transform(dataframe['label'])

# Separate features and labels
features = dataframe.iloc[:, 1:-1].values
labels = dataframe.iloc[:, -1].values

# Split the dataset into training, validation, and testing sets
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
# val_features, test_features, val_labels, test_labels = train_test_split(test_features, test_labels, test_size=0.5, random_state=42)

train_features, remaining_features, train_labels, remaining_labels = train_test_split(features, labels, test_size=0.1, random_state=42)
val_features, test_features, val_labels, test_labels = train_test_split(remaining_features, remaining_labels, test_size=0.5, random_state=42)

# Normalize the feature data
max_value = np.max(train_features)
train_features = train_features / max_value
val_features = val_features / max_value
test_features = test_features / max_value

# Defining a simple neural network model suitable for tabular data
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(train_features.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(np.unique(labels)), activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(train_features, train_labels, epochs=10, batch_size=32,
          validation_data=(val_features, val_labels))

# Plotting data
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('loss_plot.png')
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig('accuracy_plot.png')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=2)
print(f"\n\n\n\nTest Accuracy: {test_acc}, Test Loss: {test_loss}\n\n\n\n")    
