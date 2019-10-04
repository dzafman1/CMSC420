#model = Sequential()
#
#model.add(Dense(200, input_dim=X_train_scaled.shape[1], init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='linear'))
#
#
#sgd = SGD(lr=0.01)
#model.compile(optimizer=sgd,loss='mean_squared_error')
#
#hist = model.fit(X_train_scaled.as_matrix(), y_train.as_matrix(), nb_epoch=300,
#verbose=1, validation_data=(X_test_scaled.as_matrix(), y_test.as_matrix()))
#
#print(hist.history.keys())
#
## summarize history for loss
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title('model loss relu')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.axis([0, 300, 0, 20])
#plt.show()


def testKerasSequentialModelWithSGDOptimizer(df):
  from keras.models import Sequential
  from keras.layers import Dense, Activation
  from keras.optimizers import SGD
  from sklearn import preprocessing
  from pandas.api.types import is_numeric_dtype
  import pandas

  quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
  numDf = df[quantitativeColumns]
  X_scaled = preprocessing.scale(numDf[quantitativeColumns[:-1]].values)
  y = numDf[[quantitativeColumns[-1]]].values

  model = Sequential()

  model.add(Dense(200, input_dim=X_scaled.shape[1], init='uniform', activation='relu'))
  model.add(Dense(1, init='uniform', activation='linear'))

  sgd = SGD(lr=0.01)
  model.compile(optimizer=sgd,loss='mean_squared_error')

  hist = model.fit(X_scaled, y, nb_epoch=300, verbose=1, validation_split=0.2)
  return pandas.DataFrame(hist.history)
