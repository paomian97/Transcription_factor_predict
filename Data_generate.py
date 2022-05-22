import numpy as np

# Read positive sample data from the training set
def Load_train_positive():
    with open(file=Path_train_Positive, mode='r') as file:
        file = file.read()
        data = file.strip().split('\n')
        size = len(data)
        seqs_name = []
        seqs = []
        for i in range(0, size, 2):
            seqs_name.append(data[i])
            seqs.append(data[i+1])
        label = np.ones(len(seqs))
        np.save(r'./Dataset_preprocess/Train_seq_positive', seqs)
        np.save(r'./Dataset_preprocess/Train_label_positive', label)
    return seqs, label

# Read negative sample data from the training set
def Load_train_negative():
    with open(file=Path_train_Negative, mode='r') as file:
        file = file.read()
        data = file.strip().split('\n')
        size = len(data)
        seqs_name = []
        seqs = []
        for i in range(0, size, 2):
            seqs_name.append(data[i])
            seqs.append(data[i+1])
        label = np.zeros(len(seqs))
        np.save(r'./Dataset_preprocess/Train_seq_negative', seqs)
        np.save(r'./Dataset_preprocess/Train_label_negative', label)
    return seqs, label

# Read positive sample data from the testing set
def Load_test_positive():
    with open(file=Path_test_Positive, mode='r') as file:
        file = file.read()
        data = file.strip().split('\n')
        size = len(data)
        seqs_name = []
        seqs = []
        for i in range(0, size, 2):
            seqs_name.append(data[i])
            seqs.append(data[i+1])
        label = np.ones(len(seqs))
        np.save(r'./Dataset_preprocess/Test_seq_positive', seqs)
        np.save(r'./Dataset_preprocess/Test_label_positive', label)
    return seqs, label

# Read negative sample data from the testing set
def Load_test_negative():
    with open(file=Path_test_Negative, mode='r') as file:
        file = file.read()
        data = file.strip().split('\n')
        size = len(data)
        seqs_name = []
        seqs = []
        for i in range(0, size, 2):
            seqs_name.append(data[i])
            seqs.append(data[i+1])
        label = np.zeros(len(seqs))
        np.save(r'./Dataset_preprocess/Test_seq_negative', seqs)
        np.save(r'./Dataset_preprocess/Test_label_negative', label)
    return seqs, label

if __name__ == '__main__':
    Path_train_Positive = r'./Dataset/train_TFs.txt'
    Path_train_Negative = r'./Dataset/train_NTFs.txt'
    Path_test_Positive = r'./Dataset/test_TFs.txt'
    Path_test_Negative = r'./Dataset/test_NTFs.txt'

    train_seq_positive, train_label_positive = Load_train_positive()
    train_seq_negative, train_label_negative = Load_train_negative()
    test_seq_positive, test_label_positive = Load_test_positive()
    test_seq_negative, test_label_negative = Load_test_negative()

    print('The data is loaded and has been saved in the Dataset_preprocess')
    print('Data size:')
    print('train_seq_positive,train_label_positive:', len(train_seq_positive), len(train_label_positive))
    print('train_seq_negative,train_label_negative:', len(train_seq_negative), len(train_label_negative))
    print('test_seq_positive,test_label_positive:', len(test_seq_positive), len(test_label_positive))
    print('test_seq_negative,test_label_negative:', len(test_seq_negative), len(test_label_negative))
