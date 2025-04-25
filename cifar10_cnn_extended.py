import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Data augmentation to improve model generalization
datagen = ImageDataGenerator(
    rotation_range=15,        # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,    # Randomly shift images horizontally by 10%
    height_shift_range=0.1,   # Randomly shift images vertically by 10%
    horizontal_flip=True,      # Randomly flip images horizontally
    zoom_range=0.1            # Randomly zoom in/out by 10%
)
datagen.fit(x_train)

# Build improved CNN model
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    layers.BatchNormalization(),  # Normalize activations to stabilize training
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),         # Dropout to prevent overfitting

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Dense layers for classification
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Stop when validation loss stops decreasing
    patience=5,          # Wait 5 epochs before stopping
    restore_best_weights=True  # Restore weights from best epoch
)

# Train model with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=50,  # Increased epochs, early stopping will halt if needed
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping])

# Evaluate model on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_plots.png')
plt.close()

# Function to plot sample predictions
def plot_sample_predictions(images, true_labels, predictions, class_names, num_samples=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        pred_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[true_labels[i][0]]
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=10)
        plt.axis('off')
    plt.savefig('sample_predictions.png')
    plt.close()

# Predict on test images
sample_images = x_test[:10]
sample_true_labels = y_test[:10]
predictions = model.predict(sample_images)

# Plot sample predictions
plot_sample_predictions(sample_images, sample_true_labels, predictions, class_names)
print("Sample predictions plotted and saved as 'sample_predictions.png'")

# Generate and plot confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix plotted and saved as 'confusion_matrix.png'")