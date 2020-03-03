import tensorflow as tf
import pandas as pd
import numpy as np
import datetime


from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import model
import datagen
from datagen import SampleType


def smooth_l1_loss(y_true, y_pred, sigma=1.0):
    sigma2 = sigma ** 2
    thresold = 1 / sigma

    abs_error = tf.abs(y_true - y_pred)
    loss_smaller = 0.5 * sigma2 * tf.square(abs_error)
    loss_greater = abs_error - 0.5 / sigma2
    loss = tf.where(abs_error < thresold, loss_smaller, loss_greater)
    return tf.reduce_mean(loss)


def face_cls_filter(y_true, y_pred):
    label = y_true[:, 0]
    mask = tf.logical_or(
        tf.equal(label, SampleType.positive.value),
        tf.equal(label, SampleType.negative.value)
    )
    cls_true = tf.boolean_mask(y_true, mask)[:, 1:]
    cls_pred = tf.boolean_mask(y_pred, mask)[:, 1:]
    return cls_true, cls_pred


def bbox_reg_filter(y_true, y_pred):
    label = y_true[:, 0]
    mask = tf.logical_or(
        tf.equal(label, SampleType.positive.value),
        tf.equal(label, SampleType.partial.value)
    )
    bbox_reg_true = tf.boolean_mask(y_true, mask)[:, 1:]
    bbox_reg_pred = tf.boolean_mask(y_pred, mask)[:, 1:]
    return bbox_reg_true, bbox_reg_pred


def ldmk_reg_filter(y_true, y_pred):
    label = y_true[:, 0]
    mask = tf.equal(label, SampleType.landmark.value)
    ldmk_reg_true = tf.boolean_mask(y_true, mask)[:, 1:]
    ldmk_reg_pred = tf.boolean_mask(y_pred, mask)[:, 1:]
    return ldmk_reg_true, ldmk_reg_pred


def face_cls_loss(y_true, y_pred):
    cls_true, cls_pred = face_cls_filter(y_true, y_pred)
    cls_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=cls_pred, labels=cls_true)
    loss = tf.reduce_mean(cls_loss)
    return loss


def bbox_reg_mse_loss(y_true, y_pred):
    bbox_reg_true, bbox_reg_pred = bbox_reg_filter(y_true, y_pred)
    bbox_reg_loss = tf.losses.mean_squared_error(bbox_reg_true, bbox_reg_pred)
    return bbox_reg_loss


def bbox_reg_smooth_l1_loss(y_true, y_pred):
    bbox_reg_true, bbox_reg_pred = bbox_reg_filter(y_true, y_pred)
    bbox_reg_loss = smooth_l1_loss(bbox_reg_true, bbox_reg_pred)
    return bbox_reg_loss


def ldmk_reg_mse_loss(y_true, y_pred):
    ldmk_reg_true, ldmk_reg_pred = ldmk_reg_filter(y_true, y_pred)
    ldmk_reg_loss = tf.losses.mean_squared_error(ldmk_reg_true, ldmk_reg_pred)
    return ldmk_reg_loss


def ldmk_reg_smooth_l1_loss(y_true, y_pred):
    ldmk_reg_true, ldmk_reg_pred = ldmk_reg_filter(y_true, y_pred)
    ldmk_reg_loss = smooth_l1_loss(ldmk_reg_true, ldmk_reg_pred)
    return ldmk_reg_loss


def face_cls_metric_activation(y_true, y_pred):
    y_true, y_pred = face_cls_filter(y_true, y_pred)
    y_pred = tf.nn.softmax(y_pred)
    cls_true = tf.argmax(y_true, axis=-1)
    cls_pred = tf.argmax(y_pred, axis=-1)
    return cls_true, cls_pred


def accuracy(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)
    right_predictions = tf.cast(tf.equal(cls_true, cls_pred), tf.int32)
    accuracy = tf.reduce_sum(right_predictions) / tf.size(right_predictions)
    return accuracy


def recall(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)
    true_positives = tf.cast(tf.reduce_sum(cls_true * cls_pred), tf.float32)
    possible_positives = tf.cast(tf.reduce_sum(cls_true), tf.float32)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)
    true_positives = tf.cast(tf.reduce_sum(cls_true * cls_pred), tf.float32)
    predicted_positives = tf.cast(tf.reduce_sum(cls_pred), tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)

    true_positives = tf.cast(tf.reduce_sum(cls_true * cls_pred), tf.float32)
    possible_positives = tf.cast(tf.reduce_sum(cls_true), tf.float32)
    predicted_positives = tf.cast(tf.reduce_sum(cls_pred), tf.float32)

    recall = true_positives / (possible_positives + K.epsilon())
    precision = true_positives / (predicted_positives + K.epsilon())
    return 2 * precision * recall / (precision + recall + K.epsilon())


if __name__ == "__main__":

    net = 'p'

    if net == 'o':
        model = model.make_onet()
        loss = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_smooth_l1_loss,
                'ldmk_reg': ldmk_reg_smooth_l1_loss}
        loss_weights = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 1.0}
        metrics = {'face_cls': [precision, f1_score, accuracy]}
        net_input_size = 48
        saved_name = "model_onet.h5"
    elif net == 'p':
        model = model.make_pnet()
        loss = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_mse_loss,
                'ldmk_reg': ldmk_reg_mse_loss}
        loss_weights = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 0.5}
        metrics = {'face_cls': [recall, accuracy]}
        net_input_size = 12
        saved_name = "model_pnet.h5"
    elif net == 'r':
        model = model.make_rnet()
        loss = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_smooth_l1_loss,
                'ldmk_reg': ldmk_reg_smooth_l1_loss}
        loss_weights = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 0.5}
        metrics = {'face_cls': [precision, f1_score, accuracy]}
        net_input_size = 24
        saved_name = "model_rnet.h5"

    # model.load_weights('test_model_onet.h5')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),  # Optimizer
                  loss=loss,
                  #   loss=tf.keras.losses.MSE,
                  loss_weights=loss_weights,
                  metrics=metrics)
# datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir)
    #

    datagener = datagen.augmented_data_generator(
        net_input_size, net_input_size)

    history = model.fit(datagener, epochs=10, use_multiprocessing=False, workers=0,
                        steps_per_epoch=100, shuffle=True, callbacks=[tensorboard_callback])

    print('\nhistory dict:', history.history)
    model.save(saved_name)

    pass
