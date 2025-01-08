import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import drive

# Mount Google Drive for dataset access
drive.mount('/content/drive')

# Paths to dataset folders
train_dir = '/content/drive/My Drive/Traffic_Signs/train'
valid_dir = '/content/drive/My Drive/Traffic_Signs/valid'
test_dir = '/content/drive/My Drive/Traffic_Signs/test'

# Load datasets with preprocessing
train = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(244, 244),
    batch_size=32
)

valid = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir,
    image_size=(244, 244),
    batch_size=32
)

test = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(244, 244),
    batch_size=32
)

# Display class names in the dataset
print("Class Names:", train.class_names)

# Visualize a few images from the training dataset
for images, labels in train.take(1):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.ravel()

    for i in range(12):
        axes[i].imshow(images[i].numpy().astype('uint8'))  # Display the image
        axes[i].set_title(f"Label: {labels[i].numpy()}")  # Display the label
        axes[i].axis('off')  # Remove axes for better visualization

    plt.tight_layout()
    plt.show()

# Normalize the datasets (scale pixel values between 0 and 1)
train = train.map(lambda x, y: (x / 255.0, y))
valid = valid.map(lambda x, y: (x / 255.0, y))
test = test.map(lambda x, y: (x / 255.0, y))

# Display the shape of images and labels for verification
for images, labels in train.take(1):
    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)

# Define a CNN model
from tensorflow.keras import models, layers

CNN = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(244, 244, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(43, activation='softmax')  # Assuming 43 traffic sign classes
])

# Display the model architecture
CNN.summary()

# Compile the model
CNN.compile(optimizer='Adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])

# Configure EarlyStopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = CNN.fit(
    train,
    epochs=20,
    validation_data=valid,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model on the test dataset
score = CNN.evaluate(test)
print(f'Test Accuracy: {score[1]}')

# Extract accuracy and loss metrics
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation loss
epochs = range(len(accuracy))
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'ro-', label='Training Loss')
plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict labels for the test dataset
predicted_classes = np.concatenate([
    CNN.predict(images).argmax(axis=1) for images, _ in test
], axis=0)

# Extract true labels
y_true = np.concatenate([labels.numpy() for _, labels in test], axis=0)

# Generate and visualize confusion matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_true, predicted_classes))

# Visualize test images with predicted labels
L, W = 5, 5  # Grid dimensions
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    for images, labels in test.take(1):
        axes[i].imshow(images[i].numpy().astype('uint8'))
        axes[i].set_title(f"Pred: {predicted_classes[i]}")
        axes[i].axis('off')

plt.subplots_adjust(wspace=1)
plt.show()

#Save the model
CNN.save('traffic_sign_model.h5')