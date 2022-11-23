import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed
from random import randint
import tensorflow as tf
from PIL import Image
from pathlib import Path
import pathlib
import pandas as pd
from IPython.core.display import HTML

from src.sormani.system import STORAGE_DL

BATCH_SIZE = 32
IMG_SIZE = (7500, 600)

class cnn:

  def __init__(self):
    self.train_dir = os.path.join(STORAGE_DL, 'train')
    self.validation_dir = os.path.join(STORAGE_DL, 'validation')
    self.train_dataset = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                                     shuffle=True,
                                                                     batch_size=BATCH_SIZE,
                                                                     image_size=IMG_SIZE)
    self.validation_dataset = tf.keras.utils.image_dataset_from_directory(self.validation_dir,
                                                                          shuffle=True,
                                                                          batch_size=BATCH_SIZE,
                                                                          image_size=IMG_SIZE)
    self.class_names = self.train_dataset.class_names

  def exec_cnn(self):
    plt.figure(figsize=(10, 10))
    for images, labels in self.train_dataset.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(self.class_names[labels[i]])
        plt.axis("off")

    val_batches = tf.data.experimental.cardinality(self.validation_dataset)
    test_dataset = self.validation_dataset.take(val_batches // 5)
    self.validation_dataset = self.validation_dataset.skip(val_batches // 5)

    print('Number of validation batches: %d' % tf.data.experimental.cardinality(self.validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    AUTOTUNE = tf.data.AUTOTUNE

    self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
    self.validation_dataset = self.validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])

    for image, _ in self.train_dataset.take(1):
      plt.figure(figsize=(10, 10))
      first_image = image[0]
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(self.train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False

    # Let's take a look at the base model architecture
    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)


    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    len(model.trainable_variables)

    initial_epochs = 10

    loss0, accuracy0 = model.evaluate(self.validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(self.train_dataset,
                        epochs=initial_epochs,
                        validation_data=self.validation_dataset)


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                  metrics=['accuracy'])

    model.summary()

    len(model.trainable_variables)

    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(self.train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=self.validation_dataset)

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(image_batch[i].astype("uint8"))
      plt.title(self.class_names[predictions[i]])
      plt.axis("off")

def prepare_png():
  image_path = os.path.join(STORAGE_DL, 'all')
  train_path = os.path.join(STORAGE_DL, 'train')
  for filedir, dirs, files in os.walk(image_path):
    files.sort()
    for file in files:
      image = Image.open(os.path.join(image_path, file))
      file_name = Path(file).stem + '.png'
      image.save(os.path.join(train_path, file_name), 'PNG', quality=100)
  pass

def prepare_cnn(validation = 0.1, test = 0.1):
  seed(28362)
  train_path = os.path.join(STORAGE_DL, 'train')
  validation_path = os.path.join(STORAGE_DL, 'validation')
  test_path = os.path.join(STORAGE_DL, 'test')
  for filedir, dirs, files in os.walk(train_path):
    l = len(files)
    m = l
    for i in range(int(l * validation)):
      j = randint(0, m)
      file = files[j]
      file_name = Path(file).stem + '.png'
      files.remove(file)
      os.rename(os.path.join(train_path, file_name), os.path.join(validation_path, file_name))
      m -= 1
  for filedir, dirs, files in os.walk(train_path):
    m = len(files) - 1
    for i in range(int(l * test)):
      j = randint(0, m)
      file = files[j]
      file_name = Path(file).stem + '.png'
      files.remove(file)
      os.rename(os.path.join(train_path, file_name), os.path.join(test_path, file_name))
      m -= 1
  pass

def move_to_train():
  train_path = os.path.join(STORAGE_DL, 'train')
  validation_path = os.path.join(STORAGE_DL, 'validation')
  test_path = os.path.join(STORAGE_DL, 'test')
  for filedir, dirs, files in os.walk(validation_path):
    for file in files:
      file_name = Path(file).stem + '.png'
      os.rename(os.path.join(validation_path, file_name), os.path.join(train_path, file_name))
  for filedir, dirs, files in os.walk(test_path):
    for file in files:
      file_name = Path(file).stem + '.png'
      os.rename(os.path.join(test_path, file_name), os.path.join(train_path, file_name))

def path_to_image_html(path):
  return '<img src="' + path + '" width="60" >'

def see_images():
  for filedir, dirs, files in os.walk(STORAGE_DL):
    for file in files:
      files = files[:10]
      df = pd.DataFrame(files, columns=['Page'])
      df.to_html(escape=False, formatters=dict(Page=path_to_image_html))
      HTML(df.to_html(escape=False, formatters=dict(Country=path_to_image_html)))
      return

