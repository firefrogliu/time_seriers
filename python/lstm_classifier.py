
from keras import backend as K
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def lstm_classfier(model_disc: Lstm_model_disc):
    window, predict_hour, test, trainX, trainY, testX, testY, train_size = prepare_classification_data(model_disc)
    epoch = model_disc.epcoh
    print('training on', model_disc.model_name)
    numpy.random.seed(7)
    # create and fit the LSTM network
    model = Sequential()
    model.add(CuDNNLSTM(window, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(window/2)))
    model.add(Dense(int(window/4)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision_m])
    model.fit(trainX, trainY, epochs= epoch, batch_size= 256, verbose=2)
    model.save(MODEL_PATH + model_disc.site + model_disc.model_name  +'.h5')
    scores = model.evaluate(testX, testY, verbose=0)
    
    testPredict = model.predict(testX)
    # plt.clf()
    # plt.plot(test[:,0])    
    # plt.plot(testPredict)
    
    #plt.show()
