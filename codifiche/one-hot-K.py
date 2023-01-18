import numpy as np

def loadMonk(file_, filetype, encodeLabel=False):
    filename="/Users/gabrielebenanti/Documents/ML_project/MONK/monks-{}.{}".format(file_, filetype) #da cambiare a seconda del computer
    def encode(vector, label=False):
        if label:
            twoFeatures = {'0': [1, 0], '1': [0, 1]}
            return twoFeatures[str(vector)]
        else:
            retVector=[]
            twoFeatures_output={'0':[0], '1':[1]}
            twoFeatures={'1':[1,0], '2':[0,1]}
            threeFeatures={'1':[1,0,0],'2':[0,1,0],'3':[0,0,1]}
            fourFeatures={'1':[1,0,0,0],'2':[0,1,0,0],'3':[0,0,1,0],'4':[0,0,0,1]}
            encodingDict={
                '0':twoFeatures_output,
                '1':threeFeatures,
                '2':threeFeatures,
                '3':twoFeatures,
                '4':threeFeatures,
                '5':fourFeatures,
                '6':twoFeatures
            }
            for idx, val in enumerate(vector):
                retVector.extend(encodingDict[str(idx)][str(val)])
            return retVector

    with open(filename) as f:
        data_=[]
        for line in f.readlines():
            rows=[x for x in line.split(' ')][1:-1]
            temp=encode(rows)
            assert len(temp)==18
            data_.append(temp)
        data_=np.array(data_, dtype='float16')

    return data_

trainData=loadMonk(1, 'train', encodeLabel=False)
np.savetxt('./MONK/monk_1_train_hot.csv', trainData, delimiter=',')
trainData = loadMonk(1, 'test', encodeLabel=False)
np.savetxt('./MONK/monk_1_test_hot.csv', trainData, delimiter=',')
trainData = loadMonk(2, 'train', encodeLabel=False)
np.savetxt('./MONK/monk_2_train_hot.csv',  trainData, delimiter=',')
trainData = loadMonk(2, 'test', encodeLabel=False)
np.savetxt('./MONK/monk_2_test_hot.csv',  trainData, delimiter=',')
trainData = loadMonk(3, 'train', encodeLabel=False)
np.savetxt('./MONK/monk_3_train_hot.csv',  trainData, delimiter=',')
trainData = loadMonk(3, 'test', encodeLabel=False)
np.savetxt('./MONK/monk_3_test_hot.csv',  trainData, delimiter=',')