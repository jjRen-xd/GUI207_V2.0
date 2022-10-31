'''神经网络特征提取'''

import tensorflow.keras as keras


def net_fea_extract(path, train_x, train_y, test_x, test_y, epoch, batch_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(16, 8, strides=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.Conv1D(32, 4, strides=2, activation='relu', padding="valid"))
    model.add(keras.layers.MaxPooling1D(2))

    # model.add(keras.layers.Conv1D(64, 4, strides=2, activation='relu', padding="valid"))
    # model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.Conv1D(32, 2, strides=1, activation='relu', padding="valid"))
    model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    learn_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='lr', factor=0.99, patience=3,
                                                             verbose=0, min_lr=0.0001)
    checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learn_rate_reduction]
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, shuffle=True,
              validation_data=(test_x, test_y), callbacks=callbacks_list, verbose=0, validation_freq=10)
    test_model = keras.models.load_model(path)
    functor = keras.models.Model(inputs=test_model.input, outputs=test_model.layers[-2].output)  # 输出模型倒数第二层
    train_fea = functor.predict(train_x)
    test_fea = functor.predict(test_x)

    return train_fea, test_fea
