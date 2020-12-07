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
from keras.models import Model
import mltools,dataset2016
import MCLDNN as mcl
import argparse

def predict(model,batch,data,classes):
    # Plot confusion matrix
    test_Y_hat = model.predict([X_test,X1_test,X2_test], batch_size=batch)
    confnorm,_,_ = mltools.calculate_confusion_matrix(Y_test,test_Y_hat,classes)
    mltools.plot_confusion_matrix(confnorm, labels=classes,save_filename='figure/mclstm_total_confusion.png')
    
    # Plot confusion matrix
    acc = {}
    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )
    i = 0
    for snr in snrs:
        
        # Extract classes @ SNR
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X1_i = X1_test[np.where(np.array(test_SNRs) == snr)]
        test_X2_i = X2_test[np.where(np.array(test_SNRs) == snr)]
        test_X_i=X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        
        # Estimate classes
        test_Y_i_hat = model.predict([test_X_i,test_X1_i,test_X2_i])
        confnorm_i,cor,ncor = mltools.calculate_confusion_matrix(test_Y_i,test_Y_i_hat,classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        mltools.plot_confusion_matrix(confnorm_i, labels=classes, title="Confusion Matrix" ,save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr,100.0*acc[snr]))
        acc_mod_snr[:,i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i,axis=1),3)
        i = i +1
    
    # Save acc for mod per SNR
    fd = open('predictresult/acc_for_mod.dat', 'wb')
    pickle.dump((acc_mod_snr), fd)
    fd.close()

    # Save results to a pickle file for plotting later
    print(acc)
    fd = open('predictresult/acc.dat','wb')
    pickle.dump( (acc) , fd )

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title(" Classification Accuracy on RadioML")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')

if __name__ == "__main__":
    # Set up some params
    parser = argparse.ArgumentParser(description="MCLDNN_test")
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size")
    parser.add_argument("--filepath", type=str, default='./weights.h5', help='Path for saving and reloading the weight')
    parser.add_argument("--datasetpath", type=str, default='./RML2016.10a_dict.pkl', help='Path for the dataset')
    parser.add_argument("--data", type=int, default=0, help='Select the RadioML2016.10a or RadioML2016.10b, 0 or 1')
    opt = parser.parse_args()

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
    classes = mods

    # Build framework (model)
    if opt.data==0:
        model = mcl.MCLDNN(classes=11)
    elif opt.data==1:
        model = mcl.MCLDNN(classes=10)
    else:
        print('use correct data number: 0 or 1')

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    # We re-load the best weights
    model.load_weights(opt.filepath)

    predict(model,opt.batch_size,opt.data,classes)

    