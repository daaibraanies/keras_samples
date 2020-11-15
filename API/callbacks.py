import keras
import numpy as np

callbacks_list = [
    ##Stop learning of val acc doesnt change for 2 epochs
    keras.callbacks.EarlyStopping(
        monitor = 'val_acc',
        patience = 1
    ),
    ##Saves weights each time val_loss has improved
    keras.callbacks.ModelCheckpoint(
        filepath = 'my_model.h5',
        monitor = 'val_loss',
        save_best_only = True
    ),
    ##Reduces learning rate 10 times if val_loss doesnt change for 11 epochs
    keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = 10
    )
]


#Implementing keras.callbacks.Callback to create callback function
#on_epoch_begin / on_epich_end
#on_batch_begin / on batch_end
#on_train_begin / on_train_end

#Custom callback class
#Each end of epoch it saves file with layers' activations
class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input,layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError("Validation data is required.")

        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activation_epoch_'+str(epoch)+'.npz','wb')
        np.savez(f,activations)
        f.close()