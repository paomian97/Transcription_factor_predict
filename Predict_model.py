import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

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

if __name__ == '__main__':
    base = 'XACDEFGHIKLMNPQRSTVWY'
    Protein = [a for a in base]
    Protein_dict = {x: y for y, x in enumerate(Protein)}

    # Load data
    test_seq_positive = np.load(r'./Dataset_preprocess/Test_seq_positive.npy')
    test_seq_negative = np.load(r'./Dataset_preprocess/Test_seq_negative.npy')
    test_label_positive = np.load(r'./Dataset_preprocess/Test_label_positive.npy')
    test_label_negative = np.load(r'./Dataset_preprocess/Test_label_negative.npy')

    test_seq = np.concatenate([test_seq_positive, test_seq_negative], axis=0)
    test_label = np.concatenate([test_label_positive, test_label_negative], axis=0)

    predict_outcome = model_predict(test_seq)

    print('Predicted results of the test set:')
    print('Accuracy: ', accuracy_score(test_label, predict_outcome))

