import os
from keras.models import Model
from keras.layers import Input,Dense,Conv1D,Dropout,concatenate,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM

def MCLDNN(weights=None,
             input_shape1=[2,128],
             input_shape2=[128,1],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr=0.5

    input1=Input(input_shape1+[1],name='I/Qchannel')
    input2=Input(input_shape2,name='Ichannel')
    input3=Input(input_shape2,name='Qchannel')

    # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
    x1=Conv2D(50,(2,8),padding='same',activation="relu",name="Conv1",kernel_initializer="glorot_uniform")(input1)
    x2=Conv1D(50,8,padding='causal',activation="relu",name="Conv2",kernel_initializer="glorot_uniform")(input2)
    x2_reshape=Reshape([-1,128,50])(x2)
    x3=Conv1D(50,8,padding='causal',activation="relu",name="Conv3",kernel_initializer="glorot_uniform")(input3)
    x3_reshape=Reshape([-1,128,50],name="reshap2")(x3)
    x=concatenate([x2_reshape,x3_reshape],axis=1,name='Concatenate1')
    x=Conv2D(50,(1,8), padding='same',activation="relu",name="Conv4",kernel_initializer="glorot_uniform")(x)
    x=concatenate([x1,x],name="Concatenate2")
    x=Conv2D(100,(2,5),padding="valid",activation="relu",name="Conv5",kernel_initializer="glorot_uniform")(x)
    
    # Part-B: TRemporal Characteristics Extraction Section
    x=Reshape(target_shape=((124,100)))(x)
    x=CuDNNLSTM(units=128,return_sequences=True,name="LSTM1")(x)
    x=CuDNNLSTM(units=128,name="LSTM2")(x)

    #DNN
    x=Dense(128,activation="selu",name="FC1")(x)
    x=Dropout(dr)(x)
    x=Dense(128,activation="selu",name="FC2")(x)
    x=Dropout(dr)(x)
    x=Dense(classes,activation="softmax",name="Softmax")(x)

    model=Model(inputs=[input1,input2,input3],outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)
    
    return model

import keras
from keras.optimizers import adam
if __name__ == '__main__':
    # for the RaioML2016.10a dataset
    model = MCLDNN(classes=11)

    # for the RadioML2016.10b dataset
    # model = MCLDNN(classes=10)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()