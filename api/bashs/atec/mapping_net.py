'特征适应变换'

import tensorflow.keras as keras
from tensorflow.keras import regularizers


# 特征映射
def fea_mapping(path, train_x, train_y, test_x, test_y, epoch, batch_size):
    fitting_model = keras.Sequential([
        keras.layers.Conv1D(30, kernel_size=5, padding='valid', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool1D(pool_size=2, strides=2),

        keras.layers.Conv1D(25, kernel_size=5, padding='valid', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool1D(pool_size=2, strides=2),

        keras.layers.Conv1D(15, kernel_size=5, padding='valid', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool1D(pool_size=2, strides=2),

        keras.layers.Flatten(),

        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='softmax'),
        keras.layers.Dense(len(train_y[0]))
    ])
    fitting_model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='lr', factor=0.99, patience=3,
                                                             verbose=0, min_lr=0.00001)
    checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='val_loss', verbose=0,
                                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint, learn_rate_reduction]
    fitting_model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, validation_data=(test_x, test_y),
                      callbacks=callbacks_list, verbose=0, validation_freq=10)
    test_model = keras.models.load_model(path)
    train_pred = test_model.predict(train_x)
    test_pred = test_model.predict(test_x)

    return train_pred, test_pred
