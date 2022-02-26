#import packages
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

#visualize fake images
def plot_results(images, n_cols=None):
    display.clear_output(wait=False)
    n_cols = n_cols or len(images)
    n_rows = (len(images)-1)//n_cols + 1
    
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    
    plt.figure(figsize = (n_cols, n_rows))
    
    for index, image in enumerate(images):
        plt.suplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap = "binary")
        plt.axis("off")

#download training images
(X_train, _), _ = keras.datasets.fashion_mnist.load_data()

#normalize the pixel values
X_train = X_train.astype(np.float32) / 255

#reshape and rescale
X_train = X_train.reshape(-1, 28, 28, 1) * 2. - 1.
BATCH_SIZE = 128

#create batches of tensors to be fed into the model
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)

#Create Generator
codings_size = 32
generator = keras.models.Sequential([
    keras.layers.Dense(7*7*128, input_shape=[codings_size]),
    keras.layers.Reshape([7,7,128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME", activation="tanh")])

generator.summary()

#generate a batch of noise input(batch size = 16)
test_noise = tf.random.normal([16, codings_size])
#feed the batch to the untrained generator
test_image = generator(test_noise)
#visualize the sample output
plot_results(test_image, n_cols=4)
print(f'shape of the generated batch: {test_image.shape}')

#Create Discriminator
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                        activation = keras.layers.LeakyReLU(0.2),
                        input_shape = [28,28,1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                        activation = keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")])

discriminator.summary()

gan = keras.models.Sequential([generator, discriminator])

#configure model for training
discriminator.compile(loss = "binary_crossentropy", optimizer = "rmsprop")
discriminator.trainable = False
gan.compile(loss = "binary_crossentropy", optimizer="rmsprop")

#Train the Model
def train_gan(gan, dataset, random_normal_dimensions, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch+1, n_epochs))
        for real_images in dataset:
            #infer batch size from the training batch
            batch_size = real_images.shape[0]
            #Train the discriminator = PHASE 1
            #create the noise
            noise = tf.random.normal(shape=[batch_size, random_normal_dimensions])
            #use noise to generate fake images
            fake_images = generator(noise)
            #create a list by concatenating the fake images with real ones
            mixed_images = tf.concat([fake_images, real_images], axis=0)
            #Create the labels for the discriminator
            #0 for fake images
            #1 for real images
            discriminator_labels = tf.concat([[0.]]*batch_size + [[1.]]*batch_size)
            #ensure that the discriminator is trainable
            discriminator.trainable = True
            #use train_on_batch to train the discriminator with the mixed images and the discriminator labels
            discriminator.train_on_batch(mixed_images, discriminator_labels)
            #Train the generator - PHASE 2
            #create a batch of noise input to feed to the GAN
            noise = tf.random.normal(shape=[batch_size, random_normal_dimensions])
            #label all generated images to be real
            generator_labels = tf.constant([[1.]]*batch_size)
            #freeze the discriminator
            discriminator.trainable = False
            #train the GAN on the noise with the labels all set to be true
            gan.train_on_batch(noise, generator_labels)
        #plot the fake images used to train the discriminator
        plot_results(fake_images, 16)
        plt.show()

train_gan(gan, dataset, codings_size, 100)