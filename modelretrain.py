"""
Preprocessing/transformation/training functions for the energy model used in
the reinforcement learning agent's environment.
"""


from keras.callbacks import ModelCheckpoint


def retrain(model, train_x, train_y, test_x, test_y, epochs=25):

    # callback to save best weights only
    checkpoint = ModelCheckpoint('weights.best.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    # do retraining
    model.fit(train_x, train_y, epochs=epochs, batch_size=16,
                   validation_data=(test_x, test_y), verbose=0, callbacks=[checkpoint])