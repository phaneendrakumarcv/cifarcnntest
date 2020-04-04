import keras
from keras.models import Sequential,save_model
from keras.layers import Convolution2D,Activation,MaxPooling2D,Dense,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import cv2
import numpy as np




training_folder = "/home/phaneendra/Downloads/cifar10-pngs-in-folders/cifar10/cifar10/train"
test_folder = "/home/phaneendra/Downloads/cifar10-pngs-in-folders/cifar10/cifar10/test"

def train():

    model = Sequential()
    model.add(Convolution2D(512,(3,3),activation='relu',input_shape=(64,64,3)))  
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Convolution2D(512,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(512,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',loss=['categorical_crossentropy'],metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            training_folder,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            test_folder,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

    model.fit_generator(train_generator,steps_per_epoch=1000,epochs=15,validation_data=validation_generator)

    model.save_weights('cifarcnn.h5')



def test():


    model = Sequential()
    model.add(Convolution2D(512,(3,3),activation='relu',input_shape=(64,64,3)))  
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Convolution2D(512,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(512,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',loss=['categorical_crossentropy'],metrics=['accuracy'])

    image = load_img("ship.jpg",target_size=(64,64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255. 
    model.load_weights('cifarcnn.h5')
    print(model.predict_classes(image))


if __name__ == "__main__":
    test()

    