import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

# Tutorial for this: https://www.youtube.com/watch?v=jztwpsIzEGc&ab_channel=NicholasRenotte

# Set tensorflow to run on my GPU (M2 Max on Apple Silicon)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Defines directory which stores the dataset
# Dataset download link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
data_dir = '../data/GTZAN/images_original'

# Makes the data pipeline so the model doesnt need to load in the entire dataset into memory
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size = (288, 432), batch_size=10).map(lambda x,y: (x/255, y)) 
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


# Define the training, testing and validation data partitions
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)



# Defining the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), 1, activation='relu', input_shape=(288, 432, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(.5))

model.add(layers.Conv2D(64, (3, 3), 1, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(.5))

model.add(layers.Conv2D(64, (3, 3), 1, activation='relu'))
model.add(layers.Dropout(.5))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

#Training the model, and storing the models in the logs/GTZAN_Classification

log_dir = 'logs/GTZAN_classification_logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
history = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

# Plotting data
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('plot.png')
plt.show()

test_loss, test_acc = model.evaluate(test, verbose=2)         
print(test_acc)    