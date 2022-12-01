import os
print(os.listdir("/Users/ryanwest/OMSCS/cs6440/final-project/data/Dataset"))
root_path = "/Users/ryanwest/OMSCS/cs6440/final-project/data/Dataset2"
TRAIN_PATH = root_path + "/Train"
VAL_PATH = root_path + "/Val"
TEST_PATH = root_path + "/Prediction"
RANDOM_PATH = root_path + "/internet_downloads"

MODEL_PATH = "/Users/ryanwest/OMSCS/cs6440/final-project/models/"


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
# import keras.utils as image
from keras.preprocessing import image
import keras.utils as image_postproccess
from datetime import datetime

#
# model = Sequential()
# model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)))
#
# model.add(Conv2D(64,(3,3),activation="relu"))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64,(3,3),activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(128,(3,3),activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(128,(3,3),activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(64,activation="relu"))
# model.add(Dropout(0.5))
#
# model.add(Dense(3,activation="softmax"))

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')
])


# model.compile(loss='categorical_crossentropy',optimizer = "adam",metrics=["accuracy"]) #sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
model.summary()


# train_datagen = image.ImageDataGenerator(
#     rescale = 1./255,
#     shear_range = 0.2,
#     zoom_range = 0.2,
#     horizontal_flip = True,
# )
train_datagen = image.ImageDataGenerator(rescale=1./255)
test_dataset = image.ImageDataGenerator(rescale = 1./255)

target_size = 227 # 224
epochs = 20
class_mode = 'categorical' #categorical
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size = (target_size,target_size),
    batch_size = 32,
    class_mode = class_mode
)

train_generator.class_indices

validation_generator = test_dataset.flow_from_directory(
    VAL_PATH,
    target_size = (target_size,target_size),
    batch_size = 32,
    class_mode = class_mode
)


hist = model.fit_generator(
    train_generator,
    steps_per_epoch = None,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = 1
)

timestamp = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
model.save(MODEL_PATH + "Detect_Covid-" + timestamp + ".h5")

# model.evaluate_generator(train_generator)
# model.evaluate_generator(validation_generator)

model = load_model(MODEL_PATH + "Detect_Covid-" + timestamp + ".h5")

y_actual = []
y_test = []
y_test_proba = []
EXAMPLE_PATH = TEST_PATH

for classif in train_generator.class_indices:
    print(classif)
    print(train_generator.class_indices[classif])
    for i in os.listdir(EXAMPLE_PATH + "/" + classif):
      img = image_postproccess.load_img(EXAMPLE_PATH + "/" + classif +"/" +i,target_size=(target_size,target_size))
      img = image_postproccess.img_to_array(img)
      img = np.expand_dims(img,axis=0)
      img = img/255
      predict_x = model.predict(img, verbose=False)
      y_test.append(np.argmax(predict_x[0]))
      y_test_proba.append(max(predict_x[0]))
      y_actual.append(train_generator.class_indices[classif])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_actual,y_test)
from sklearn.metrics import precision_recall_fscore_support
metrics_df = pd.DataFrame(precision_recall_fscore_support(y_actual, y_test, average=None),
                          columns =list(train_generator.class_indices.keys())).T
# tn, fp, fn, tp = confusion_matrix(y_actual,y_test).ravel()
# (tp + tn)/(tn+fp+fn+tp)


EXAMPLE_PATH = RANDOM_PATH

y_test = []

for i in os.listdir(EXAMPLE_PATH):
    img = image_postproccess.load_img(EXAMPLE_PATH +"/" +i,target_size=(target_size,target_size))
    img = image_postproccess.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = img/255
    predict_x = model.predict(img, verbose=False)
    y_test.append(np.argmax(predict_x[0]))
