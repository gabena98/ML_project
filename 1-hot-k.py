import numpy as np
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn import *

def loadMonk(file_, filetype):
    filename="./MONK/monks-{}.{}".format(file_, filetype)
    #leggo ogni riga del file aperto
    with open(filename) as f:
        data_=[]
        for line in f.readlines():
            temp=[x for x in line.split(' ')][1:-1]
            assert len(temp)==7
            data_.append(temp)
        data_=np.array(data_, dtype='int')
    # codifica one-hot per monk
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(data_)
    X_train = one_hot_encoder.transform(data_)
    return X_train

trainData=loadMonk(1, 'train')
np.savetxt('./MONK/monk_1_train_hot.csv', trainData, delimiter=',')
trainData=loadMonk(1, 'test')
np.savetxt('./MONK/monk_1_test_hot.csv', trainData, delimiter=',')
trainData=loadMonk(2, 'train')
np.savetxt('./MONK/monk_2_train_hot.csv', trainData, delimiter=',')
trainData=loadMonk(2, 'test')
np.savetxt('./MONK/monk_2_test_hot.csv', trainData, delimiter=',')
trainData=loadMonk(3, 'train')
np.savetxt('./MONK/monk_3_train_hot.csv', trainData, delimiter=',')
trainData=loadMonk(3, 'test')
np.savetxt('./MONK/monk_3_test_hot.csv', trainData, delimiter=',')