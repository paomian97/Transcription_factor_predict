import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Flatten, Conv1D, Bidirectional, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer
import numpy as np

class Capsule(Layer):
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 initializer='glorot_uniform',
                 activation=None,
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

        self.activation = activations.get(activation)
        self.regularizer = regularizers.get(regularizer)
        self.initializer = initializers.get(initializer)
        self.constraint = constraints.get(constraint)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1,
                                            input_dim_capsule,
                                            self.num_capsule *
                                            self.dim_capsule),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     constraint=self.constraint,
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule *
                                            self.dim_capsule),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     constraint=self.constraint,
                                     trainable=True)
        self.build = True

    def call(self, inputs):
        if self.share_weights:
            u_hat_vectors = K.conv1d(inputs, self.W)
        else:
            u_hat_vectors = K.local_conv1d(inputs, self.W, [1], [1])
        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        u_hat_vectors = K.reshape(u_hat_vectors, (batch_size,
                                                  input_num_capsule,
                                                  self.num_capsule,
                                                  self.dim_capsule))

        u_hat_vectors = K.permute_dimensions(u_hat_vectors, (0, 2, 1, 3))
        routing_weights = K.zeros_like(u_hat_vectors[:, :, :, 0])

        for i in range(self.routings):
            capsule_weights = K.softmax(routing_weights, 1)
            outputs = K.batch_dot(capsule_weights, u_hat_vectors, [2, 2])
            if K.ndim(outputs) == 4:
                outputs = K.sum(outputs, axis=1)
            if i < self.routings - 1:
                outputs = K.l2_normalize(outputs, -1)
                routing_weights = K.batch_dot(outputs, u_hat_vectors, [2, 3])
                if K.ndim(routing_weights) == 4:
                    routing_weights = K.sum(routing_weights, axis=1)

        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {'num_capsule': self.num_capsule,
                  'dim_capsule': self.dim_capsule,
                  'routings': self.routings,
                  'share_weights': self.share_weights,
                  'activation': activations.serialize(self.activation),
                  'regularizer': regularizers.serialize(self.regularizer),
                  'initializer': initializers.serialize(self.initializer),
                  'constraint': constraints.serialize(self.constraint)}
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

base = 'XACDEFGHIKLMNPQRSTVWY'
Protein = [a for a in base]
Protein_dict = {x: y for y, x in enumerate(Protein)}

def transform_encode(sequences):
    output = []
    for seq in sequences:
        tmp = []
        for AA in seq:
            tmp.append(Protein_dict[AA])
        output.append(tmp)
    return np.array(output)

def length_500(sequences):
    output = []
    for seq in sequences:
        if len(seq) > 500:
            output.append(seq[:500])
        else:
            L = len(seq)
            output.append(seq+str('X')*(500-L))
    return np.array(output)

def model_framework():
    x = Input(shape=(500))
    embedding = Embedding(input_dim=21,output_dim=10)(x)
    lstm = Bidirectional(LSTM(9, return_sequences=True))(embedding)
    capsule = Capsule(num_capsule=8,dim_capsule=6,routings=3,activation='sigmoid',share_weights=True)(lstm)
    flatten = Flatten()(capsule)
    dense_1 = Dense(27, activation='relu')(flatten)
    dropout = Dropout(0.2)(dense_1)
    dense_2 = Dense(9, activation='relu')(dropout)
    dense_3 = Dense(1, activation='sigmoid')(dense_2)

    model = tf.keras.Model(inputs=x, outputs= dense_3)
    model.compile(optimizer='Adam',
                  loss='BinaryCrossentropy',
                  metrics=['accuracy'])
    return model

def train_model(train_x,train_label):
    train_x = length_500(train_x)
    train_x_feature = transform_encode(train_x)
    model = model_framework()
    model.fit(x=train_x_feature, y=train_label, validation_split=0.2,
              batch_size=8, epochs=100, validation_freq=1)
    #model.save('./model_save')

