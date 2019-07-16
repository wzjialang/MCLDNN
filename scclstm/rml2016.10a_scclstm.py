# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#from matplotlib import pyplot as plt
import pickle, random, sys,h5py
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.regularizers import *
from keras.optimizers import adam
from keras.models import model_from_json
#from keras.utils.vis_utils import plot_model

import mltools,rmldataset2016
#import rmlmodels.CNN2Model as cnn2
#import rmlmodels.ResNetLikeModel as resnet
#import rmlmodels.VGGLikeModel as vggnet
#import rmlmodels.CLDNNLikeModel as cldnn
#import rmlmodels.SCCDNNMODEL as scd
import rmlmodels.SCCLSTM as scl

#set Keras data format as channels_first
K.set_image_data_format('channels_last')
print(K.image_data_format())

(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
    rmldataset2016.load_data()
#分别选择实部和虚部的数据集,扩展实部和虚部的数据集维度
X_train=np.expand_dims(X_train,axis=3)
X_test=np.expand_dims(X_test,axis=3)
X_val=np.expand_dims(X_val,axis=3)
X1_train=np.expand_dims(X_train[:,0,:], axis=1)
X1_test=np.expand_dims(X_test[:,0,:], axis=1)
X1_val=np.expand_dims(X_val[:,0,:],axis=1)
X2_train=np.expand_dims(X_train[:,1,:], axis=1)
X2_test=np.expand_dims(X_test[:,1,:], axis=1)
X2_val=np.expand_dims(X_val[:,1,:],axis=1)
print(X_train.shape)
print(X1_train.shape)
classes = mods


# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,2,128,1] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

# Set up some params
nb_epoch = 200     # number of epochs to train on
batch_size = 1024  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
#model = cnn2.CNN2Model(None, input_shape=in_shp,classe=len(classe))

# model.compile(loss='catagorical_crossentropy',metrics=['accuracy'],optimizer='adam')
# rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#model = vggnet.VGGLikeModel(weights=None,input_shape=[128,2], classes=11)
#model = scd.SCCDNN()
model=scl.SCCLSTM()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#plot_model(model, to_file='model_CLDNN.png',show_shapes=True) # print model
model.summary()



filepath = 'weights/weights.h5'
history = model.fit([X_train,X1_train,X2_train],
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=([X_val,X1_val,X2_val],Y_val),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=5,min_lr=0.0000001)
                #keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
                
                ]
                    )
# history = model.fit(X_train,
#     Y_train,
#     batch_size=batch_size,
#     epochs=nb_epoch,
#     verbose=1,
#     validation_data=(X_test, Y_test),
#     callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
#                     )

# we re-load the best weights once training is finished
#model.load_weights(filepath)
mltools.show_history(history)

#Show simple version of performance
score = model.evaluate([X_test,X1_test,X2_test], Y_test, verbose=1, batch_size=batch_size)
print(score)

def plot_tSNE(model,filename="data/RML2016.10a_dict.pkl"):
    from keras.models import Model
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
        rmldataset2016.load_data()
    #设计中间层输出的模型
    dense2_model = Model(inputs=model.input,outputs=model.get_layer('dense1').output)

    #提取snr下的数据进行测试
    for snr in [s for s in snrs if s > 14]:
        test_SNRs = [lbl[x][1] for x in test_idx]       #lbl: list(mod,snr)
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        #计算中间层输出
        dense2_output = dense2_model.predict(test_X_i,batch_size=32)
        Y_true = np.argmax(test_Y_i,axis=1)

        #PCA降维到50以内
        pca = PCA(n_components=50)
        dense2_output_pca = pca.fit_transform(dense2_output)

        #t-SNE降为2
        tsne = TSNE(n_components=2,perplexity=5)
        Y_sne = tsne.fit_transform(dense2_output_pca)

        fig = plt.figure(figsize = (14,12))

        # 散点图
        # plt.scatter(Y_sne[:,0],Y_sne[:,1],s=5.,color=plt.cm.Set1(Y_true / 11.))

        # 标签图
        data = Y_sne
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        for i in range(Y_sne.shape[0]):
            plt.text(data[i,0],data[i,1], str(Y_true[i]))
            plt.title('t-SNR at snr:{}'.format(snr))

        # plt.legend()  # 显示图示
        # fig.show()

def predict(model):
    # (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
    #     rmldataset2016.load_data()    
    model.load_weights(filepath)
    # Plot confusion matrix
    test_Y_hat = model.predict([X_test,X1_test,X2_test], batch_size=batch_size)
    confnorm,_,_ = mltools.calculate_confusion_matrix(Y_test,test_Y_hat,classes)
    mltools.plot_confusion_matrix(confnorm, labels=classes,save_filename='figure/total_confusion')

    # Plot confusion matrix
    acc = {}
    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )
    i = 0
    for snr in snrs:

        # extract classes @ SNR
        # test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X1_i = X1_test[np.where(np.array(test_SNRs) == snr)]
        test_X2_i = X2_test[np.where(np.array(test_SNRs) == snr)]
        test_X_i=X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict([test_X_i,test_X1_i,test_X2_i])
        confnorm_i,cor,ncor = mltools.calculate_confusion_matrix(test_Y_i,test_Y_i_hat,classes)
        acc[snr] = 1.0 * cor / (cor + ncor)

        mltools.plot_confusion_matrix(confnorm_i, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)(ACC=%2f)" % (snr,100.0*acc[snr]),save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr,100.0*acc[snr]))
        
        acc_mod_snr[:,i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i,axis=1),3)
        i = i +1
    
    #plot acc of each mod in one picture
    dis_num=11
    for g in range(int(np.ceil(acc_mod_snr.shape[0]/dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g*dis_num
        end_index = np.min([(g+1)*dis_num,acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        for i in range(beg_index,end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            # 设置数字标签
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g+1))
        plt.close()
    #save acc for mod per SNR
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
    plt.title(" Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')
    

if __name__ == '__main__':
    #plot_tSNE(model)
    predict(model)
