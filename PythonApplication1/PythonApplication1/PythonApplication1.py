#import tensorflow as tf
#import keras as K
#import numpy as np
#import matplotlib
from img2vec import img2input
import tensorflow as tf
import keras as K
import pydot
import graphviz



if __name__ == '__main__':

    X_train, X_dev, X_test, train_label, dev_label, test_label = img2input()
    
    #X_train.reshape(12127,128,128,None)

    print(X_train.shape)
    print(train_label.shape)

    model = K.Sequential()
    
    model.add(K.layers.Conv2D(4,(31,31),strides=(1,1),padding='valid',activation='relu',input_shape=(128,128,3)))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(K.layers.Conv2D(8,(16,16),strides=(1,1),padding='valid',activation='relu'))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(K.layers.Conv2D(16,(6,6),strides=(1,1),padding='valid',activation='relu'))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    
    model.add(K.layers.Dense(128,activation='relu'))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dropout(0.1))
    model.add(K.layers.Dense(64,activation='relu'))
    model.add(K.layers.Dropout(0.1))
    model.add(K.layers.Dense(16,activation='relu'))
    model.add(K.layers.Dense(1,activation = 'sigmoid'))

    #K.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


    model.fit(x=X_train, y=train_label, batch_size=32, epochs=2)

    devacc = model.evaluate(x=X_dev, y=dev_label, batch_size=32)
    testacc = model.evaluate(x=X_test, y=test_label, batch_size=32)

    print(model.metrics_names)
    print(devacc)
    print(testacc)