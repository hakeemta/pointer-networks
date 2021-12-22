import keras.backend as K


def loss(y_true, y_pred):
    # categorical_crossentropy_2rd_axis
    clipped_y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    product = y_true * K.log(clipped_y_pred)
    categorical_crossentropies = - K.sum(product, axis=2)
    return K.mean(categorical_crossentropies)


def accuracy(y_true, y_pred):
    _y_true = K.argmax(y_true, axis=2)
    _y_pred =  K.argmax(y_pred, axis=2)

    _accuracy = K.equal(_y_true, _y_pred)
    _accuracy = K.cast(_accuracy, dtype='float32')
    return K.mean(_accuracy)

