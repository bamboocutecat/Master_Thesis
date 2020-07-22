# owner Jules 曾筠筑
import tensorflow as tf
import data_loader
import numpy as np


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
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(4096, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal,
    #                                 kernel_regularizer=tf.keras.regularizers.L1L2, bias_initializer=tf.keras.initializers.RandomNormal))

    feature_vector_left = model(left_input)
    feature_vector_right = model(right_input)

    # 200719
    feature_vector_left_filter = tf.reshape(feature_vector_left, shape=(
        feature_vector_left.shape[1], feature_vector_left.shape[2], 256, 1))
    heatmap = tf.keras.backend.conv2d(
        feature_vector_right, feature_vector_left_filter,  strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1))

    
    # 200719

    # TODO  比對方式修改成捲積比對
    # L1_distance_layer = tf.keras.layers.Lambda(
    #     lambda tensor: tf.keras.backend.abs(tensor[0] - tensor[1]))
    # L1_distance = L1_distance_layer(
    #     [feature_vector_left, feature_vector_right])
    # predictions = tf.keras.layers.Dense(
    #     1, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal)(L1_distance)

    resized_heatmap = tf.keras.layers.experimental.preprocessing.Resizing(
        right_input.shape[1], right_input.shape[2])(heatmap)

    siamese_network = tf.keras.Model(
        inputs=[left_input, right_input], outputs=resized_heatmap)

    # TODO 輸出應為heatmap
    return siamese_network


model = siamese((105, 105, 1), (200, 200, 1))
model.summary()

# optimizer = tf.keras.optimizers.Adagrad(lr=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# model.train_on_batch
