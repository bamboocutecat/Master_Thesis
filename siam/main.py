# owner Jules 曾筠筑
import tensorflow as tf


def siamese(left_input_shape, right_input_shape):

    left_input = tf.keras.Input(left_input_shape)
    right_input = tf.keras.Input(right_input_shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (10, 10), activation='relu', input_shape=left_input_shape,
                                     kernel_initializer=tf.keras.initializers.RandomNormal, kernel_regularizer=tf.keras.regularizers.L1L2))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(128, (7, 7), activation='relu', input_shape=left_input_shape,
                                     kernel_initializer=tf.keras.initializers.RandomNormal, kernel_regularizer=tf.keras.regularizers.L1L2))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu', input_shape=left_input_shape,
                                     kernel_initializer=tf.keras.initializers.RandomNormal, kernel_regularizer=tf.keras.regularizers.L1L2))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(256, (4, 4), activation='relu', input_shape=left_input_shape,
                                     kernel_initializer=tf.keras.initializers.RandomNormal, kernel_regularizer=tf.keras.regularizers.L1L2))

    # TODO  修改為全捲積層
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal,
                                    kernel_regularizer=tf.keras.regularizers.L1L2, bias_initializer=tf.keras.initializers.RandomNormal))

    feature_vector_left = model(left_input)
    feature_vector_right = model(right_input)

    # TODO  比對方式修改成捲積比對
    L1_distance_layer = tf.keras.layers.Lambda(
        lambda lefttensor, righttensor: tf.keras.backend.abs(lefttensor - righttensor))

    L1_distance = L1_distance_layer(feature_vector_left, feature_vector_right)

    predictions = tf.keras.layers.Dense(
        1, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal)(L1_distance)
    siamese_network=tf.keras.Model(input=[left_input,right_input],outputs=predictions)

    # TODO 輸出應為heatmap
    return siamese_network