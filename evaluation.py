import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import random
from tensorflow.keras.utils import load_img, img_to_array

# Load the trained model
loaded_model = load_model('saved_model/reptile_classifier.h5')

# Define the directory for the validation dataset
validation_dir = "dataset/validation"

# Create a data generator for the validation set
val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Rescale pixel values

# Create a data loader for the validation set
val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),  # Adjust the size as per your model's input
    batch_size=32,
    class_mode='categorical'
)

# 1. Evaluate the model's loss and accuracy
loss, accuracy = loaded_model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 2. Confusion Matrix
# Get true labels and predictions
val_generator.reset()
y_true = val_generator.classes
y_pred = np.argmax(loaded_model.predict(val_generator), axis=-1)

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 3. Classification Report
report = classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys())
print("Classification Report:")
print(report)

# 4. Per-Class Accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(val_generator.class_indices.keys()):
    print(f"Accuracy for class '{class_name}': {class_accuracy[i] * 100:.2f}%")

# 5. Visualizing Sample Predictions
random_idx = random.randint(0, len(val_generator.filenames) - 1)
img_path = f"{validation_dir}/{val_generator.filenames[random_idx]}"
true_label = y_true[random_idx]

# Load and preprocess the image
img = load_img(img_path, target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
predicted_probabilities = loaded_model.predict(img_array)
predicted_label = np.argmax(predicted_probabilities)

# Display the image and prediction
plt.imshow(img)
plt.axis('off')
plt.title(f"True: {list(val_generator.class_indices.keys())[true_label]}\nPredicted: {list(val_generator.class_indices.keys())[predicted_label]}")
plt.show()

