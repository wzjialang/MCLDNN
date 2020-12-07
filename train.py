"Adapted from the code (https://github.com/leena201818/radioml) contributed by leena201818"
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.regularizers import *
from keras.optimizers import adam
from keras.models import model_from_json,Model
import mltools,dataset2016
import MCLDNN as mcl
import argparse


if __name__ == "__main__":
    # Set up some params
    parser = argparse.ArgumentParser(description="MCLDNN")
    parser.add_argument("--epoch", type=int, default=10000, help='Max number of training epochs')
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size")
    parser.add_argument("--filepath", type=str, default='./weights.h5', help='Path for saving and reloading the weight')
    parser.add_argument("--datasetpath", type=str, default='./RML2016.10a_dict.pkl', help='Path for the dataset')
    parser.add_argument("--data", type=int, default=0, help='Select the RadioML2016.10a or RadioML2016.10b, 0 or 1')
    opt = parser.parse_args()
    
    # Set Keras data format as channels_last
    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
        dataset2016.load_data(opt.datasetpath,opt.data)

    # Select the data set of the real part and the imaginary part, separately
    # and expand the data set dimension
    X1_train=np.expand_dims(X_train[:,0,:], axis=2)
    X1_test=np.expand_dims(X_test[:,0,:], axis=2)
    X1_val=np.expand_dims(X_val[:,0,:],axis=2)

    X2_train=np.expand_dims(X_train[:,1,:], axis=2)
    X2_test=np.expand_dims(X_test[:,1,:], axis=2)
    X2_val=np.expand_dims(X_val[:,1,:],axis=2)

    X_train=np.expand_dims(X_train,axis=3)
    X_test=np.expand_dims(X_test,axis=3)
    X_val=np.expand_dims(X_val,axis=3)
    # print(X_train.shape)
    # print(X_test.shape)

    # Build framework (model)
    if opt.data==0:
        model = mcl.MCLDNN(classes=11)
    elif opt.data==1:
        model = mcl.MCLDNN(classes=10)
    else:
        print('use correct data number: 0 or 1')

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    history = model.fit([X_train,X1_train,X2_train],
        Y_train,
        batch_size=opt.batch_size,
        epochs=opt.epoch,
        verbose=2,
        validation_data=([X_val,X1_val,X2_val],Y_val),
        callbacks = [
                    keras.callbacks.ModelCheckpoint(opt.filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=5,min_lr=0.0000001),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto'),           
                    # keras.callbacks.TensorBoard(histogram_freq=1,write_graph=True,write_images=True,batch_size=opt.batch_size)
                    ]
                        )

    # We re-load the best weights once training is finished
    model.load_weights(opt.filepath)
    mltools.show_history(history)

    # Show simple version of performance
    score = model.evaluate([X_test,X1_test,X2_test], Y_test, verbose=1, batch_size=opt.batch_size)
    print(score)


