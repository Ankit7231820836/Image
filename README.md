# Image
Image Generative AI Python Script
# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# Load and preprocess the dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 255.0  # Normalize pixel values to the range [0, 1]

# Define the generator model
generator = Sequential([
    Dense(128, input_dim=100, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Define the discriminator model
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Combine the generator and discriminator into a GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Train the GAN
epochs = 100
batch_size = 64
for epoch in range(epochs):
    for _ in range(len(X_train) // batch_size):
        # Train the discriminator
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        X = np.concatenate([real_images, fake_images])
        y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        discriminator_loss = discriminator.train_on_batch(X, y)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        y = np.ones((batch_size, 1))
        generator_loss = gan.train_on_batch(noise, y)
        
    # Print progress
    print(f'Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}')

# Generate new images
noise = np.random.normal(0, 1, (10, 100))
generated_images = generator.predict(noise)

# Display the generated images
import matplotlib.pyplot as plt
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.show()
