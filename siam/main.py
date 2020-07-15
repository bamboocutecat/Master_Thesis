# owner Jules 曾筠筑
import tensorflow as tf 


def siamese(left_input_shape,right_input_shape):

    left_input = tf.keras.Input(left_input_shape)
    right_input = tf.keras.Input(right_input_shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64,(10,10),activation='relu',input_shape=left_input_shape,kernel_initializer=tf.keras.initializers.RandomNormal,kernel_regularizer=tf.keras.regularizers.L1L2))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(128,(7,7),activation='relu',input_shape=left_input_shape,kernel_initializer=tf.keras.initializers.RandomNormal,kernel_regularizer=tf.keras.regularizers.L1L2))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(128,(4,4),activation='relu',input_shape=left_input_shape,kernel_initializer=tf.keras.initializers.RandomNormal,kernel_regularizer=tf.keras.regularizers.L1L2))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(256,(4,4),activation='relu',input_shape=left_input_shape,kernel_initializer=tf.keras.initializers.RandomNormal,kernel_regularizer=tf.keras.regularizers.L1L2))
    
