##################################################
# Script to train and validate a classification model 
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

DATASET_PATH  = 'Imageset/Only pavements'
DATASET_LOGS  = 'logs/Only pavements'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # Reducir si no hay memoria suficiente
FREEZE_LAYERS = 2  # Capas que queremos congelar durante el entrenamiento
NUM_EPOCHS    = 3
MODELO_PESOS_MEJORES = 'Models/Only pavements/best_model_resnet50.h5'

##################################################
# Step 1. Define data loading generators
##################################################
# Train data generation. Uncomment to enable data augmentation
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#                                  rotation_range=10,
#                                  width_shift_range=0.1,
#                                  height_shift_range=0.1,
#                                  shear_range=0.1,
#                                  zoom_range=0.1,
#                                  channel_shift_range=10,
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

# Validation data generator, no augmentation
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/validate',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# Test generator, no data augementation
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_batches = test_datagen.flow_from_directory(DATASET_PATH + '/test',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

##################################################
# Step 2. List classes of the batch
##################################################
print("Clases devueltas por un Batch: ")
for cls, idx in train_batches.class_indices.items():
    print('Clase #{} = {}'.format(idx, cls))

###################################################
# Step 3. RestNet50 model, using Imagenet weights 
# pretraining. Top layers not included (classification)
# No incluimos las capas de clasificación (top)
# Adding DropOut layer followed by a FC layer
# and a softmax layer
# Low learning reate as the model is already pretrained.
###################################################
modelo = ResNet50(include_top=False, weights='imagenet',
               input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = modelo.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
modelo_final = Model(inputs=modelo.input, outputs=output_layer)

# Freeze selected layers during training in order to prevent
# from training
for layer in modelo_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in modelo_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
modelo_final.compile(optimizer=Adam(learning_rate=1e-5),
          loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
modelo_final.summary()

##################################################
# Step 4. Checkpoints to record according to 
#           validation and recording of logs with
#           Tensorboard
##################################################
#filepath = 'mejor_modelo.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
filepath = MODELO_PESOS_MEJORES
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

callbacks = [checkpoint, TensorBoard(log_dir=DATASET_LOGS)]

##################################################
# Step 5. Train the model
##################################################
modelo_final.fit(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=callbacks, verbose = 1)

##################################################
# Paso 6. Validation with the best model (saved)
##################################################
#modelo = load_model(MODELO_PESOS_MEJORES)
#score = modelo.evaluate(test_batches, steps=test_batches.samples // BATCH_SIZE)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')