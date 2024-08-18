# Detection-of-Lung-Diseases

# Overview
This project involves developing a Convolutional Neural Network (CNN) to classify chest X-ray images into different lung disease categories. The goal is to create a model that can accurately identify various types of lung diseases, including bacterial pneumonia, Tuberculosis, Covid-19, Bacterial Infection.

# Features
- Data Augmentation: Applied techniques such as rescaling, width shifting, height shifting, and zooming to enhance the model’s generalization.
- CNN Architecture: Implemented a CNN model with multiple convolutional and pooling layers, followed by dense layers for classification.
- High Accuracy: Achieved high accuracy in classifying lung disease types on the test dataset.

# Prerequisites
- TensorFlow
- Keras
- Kaggle API
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`

# Usage
1. Load and preprocess the data using `ImageDataGenerator`.
2. Define and compile the CNN model:
   model = models.Sequential()
   model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
3. Train the model:
   history = model.fit(train_it, steps_per_epoch=train_it.samples // train_it.batch_size, epochs=5, validation_data=val_it)
4. Evaluate the model and save it:
   test_loss, test_accuracy = model.evaluate(test_it)
   model.save("lung_disease_classification_model.h5")
5. Load the model and make predictions:
   loaded_model = load_model("lung_disease_classification_model.h5")
   
# Results
The trained model demonstrated high accuracy in classifying different types of lung diseases. The model can be used for automated diagnosis based on chest X-ray images.