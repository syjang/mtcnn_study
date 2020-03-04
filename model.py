import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers

SampleLabelFilter = layers.Lambda(lambda x: x[..., 1:])


def make_pnet(train=True):
    if train:
        input = layers.Input(shape=[12, 12, 3])
    else:
        input = layers.Input(shape=(None, None, 3), name='Pnet_input')

    x = layers.Conv2D(10, kernel_size=(3, 3))(input)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(16, kernel_size=(3, 3))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    classifier = layers.Conv2D(
        3, 1, activation='softmax', name='face_cls_conv')(x)
    bbox_regress = layers.Conv2D(5, 1, name='bbox_reg_conv')(x)
    landmark_regress = layers.Conv2D(
        11, 1, name='ldmk_reg_conv')(x)

    if train:
        classifier = layers.Reshape((3,), name='face_cls')(classifier)
        bbox_regress = layers.Reshape((5,), name='bbox_reg')(bbox_regress)
        landmark_regress = layers.Reshape(
            (11,), name='ldmk_reg')(landmark_regress)
    else:
        classifier = SampleLabelFilter(classifier)
        classifier = layers.Activation('softmax')(classifier)
        bbox_regress = SampleLabelFilter(bbox_regress)
        landmark_regress = SampleLabelFilter(landmark_regress)

    outputs = [classifier, bbox_regress, landmark_regress]

    model = models.Model(input, outputs)
    return model


def make_rnet(train=True):
    input = layers.Input(shape=(24, 24, 3))

    x = layers.Conv2D(28, kernel_size=(3, 3))(input)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(48, kernel_size=(3, 3))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(64, kernel_size=(2, 2))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.PReLU()(x)

    classifier = layers.Dense(3, activation='softmax', name='face_cls')(x)
    bbox_regress = layers.Dense(5, name='bbox_reg')(x)
    landmark_regress = layers.Dense(11, name='ldmk_reg')(x)

    # outputs = layers.Concatenate()(
    #     [classifier, bbox_regress, landmark_regress])

    outputs = [classifier, bbox_regress, landmark_regress]
    model = models.Model(input, outputs)

    return model


def make_onet(train=True):
    input = layers.Input(shape=(48, 48, 3))

    x = layers.Conv2D(32, kernel_size=(3, 3))(input)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(64, kernel_size=(2, 2))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(128, kernel_size=(2, 2))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU()(x)

    classifier = layers.Dense(3, activation='softmax', name='face_cls')(x)
    bbox_regress = layers.Dense(5, name='bbox_reg')(x)
    landmark_regress = layers.Dense(11, name='ldmk_reg')(x)

    # outputs = layers.Concatenate()(
    #     [classifier, bbox_regress, landmark_regress])

    outputs = [classifier, bbox_regress, landmark_regress]
    # model = models.Model(input, [classifier, bbox_regress, landmark_regress])
    model = models.Model(input, outputs)
    return model


# md = make_pnet(True)
# md.summary()
# md.compile(loss=tensorflow.keras.loss.)

# md2 = make_rnet(False)
# md2.summary()


# md3 = make_onet(False)
# md3.summary()
