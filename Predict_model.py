import tensorflow as tf
import numpy as np
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

def model_predict(sequences):
    sequences = length_500(sequences)
    feature = transform_encode(sequences)

    model = tf.keras.models.load_model(r'./weight/Weight')

    predict = model.predict(feature)
    probability = (np.squeeze(predict, axis=-1))
    predict = np.int32(probability > 0.5)
    return predict
