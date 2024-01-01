##################################################
# Script to execute inference and validate a classification model 
# for concrete cracks with a ResNet-50 based model
#
# Imageset: https://www.kaggle.com/datasets/datastrophy/concrete-train-test-split-dataset?resource=download
##################################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow import constant
from PIL import Image
import numpy as np
import cv2

DATASET_PATH  = 'Imageset/Only pavements'
DATASET_LOGS  = 'logs/Only pavements'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # Reducir si no hay memoria suficiente
FREEZE_LAYERS = 2  # Capas que queremos congelar durante el entrenamiento
NUM_EPOCHS    = 3
MODELO_PESOS_MEJORES = 'Models/Only pavements/best_model_resnet50.h5'

# Test generator, no data augementation
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_batches = test_datagen.flow_from_directory(DATASET_PATH + '/test',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)


print("Clases devueltas por un Batch: ")
classes = {}
for cls, idx in test_batches.class_indices.items():
    classes[str(idx)] = cls
    print('Clase #{} = {}'.format(idx, cls))

##################################################
# Paso 6. Validation with the best model (saved)
##################################################
modelo = load_model(MODELO_PESOS_MEJORES)
#score = modelo.evaluate(test_batches, steps=test_batches.samples // BATCH_SIZE)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

##################################################
# Paso 7. Choose my own image to run inference
##################################################
# Load the image 
image_path = "Imageset/Only pavements/Selection/images.jpeg"
img = cv2.imread(image_path)
print(f'Previous shape: {img.shape}')
img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

# Convert the image to a NumPy array
img_array = np.array(img)

# Reshape the NumPy array
#reshaped_array = np.reshape(img_array, (224, 224, 3))
reshaped_array = np.expand_dims(img_array, axis=0)
print(f'New shape: {reshaped_array.shape}')

# Convert the reshaped array to a TensorFlow tensor
tfinput = constant(reshaped_array)

score = modelo.predict(tfinput)
score = np.squeeze(score)
print(f'Classes: {test_batches.class_indices}')
if score[0] > score[1]:
    print(f'This image is of class: {classes[str(0)]}')
else:
    print(f'This image is of class: {classes[str(1)]}')

cv2.imshow("Resized Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()