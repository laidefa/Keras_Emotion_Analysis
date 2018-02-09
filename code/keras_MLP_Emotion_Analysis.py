# encoding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


import keras 
import numpy as np 
from keras.datasets import imdb 


(X_train, y_train), (X_test, y_test) = imdb.load_data()


print np.reshape(X_train[0], (1, -1))

print X_train.shape
print y_train.shape


avg_len = list(map(len, X_train))
print np.mean(avg_len)



# import matplotlib.pyplot as plt 
# plt.hist(avg_len, bins=range(min(avg_len), max(avg_len) + 50, 50)) 
# plt.show()




from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence 
import keras 
import numpy as np 
from keras.datasets import imdb 
(X_train, y_train), (X_test, y_test) = imdb.load_data() 
m=max(list(map(len, X_train))+ list(map(len, X_test)))
print(m)



maxword = 400
X_train = sequence.pad_sequences(X_train, maxlen = maxword)
X_test = sequence.pad_sequences(X_test, maxlen = maxword)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1



model = Sequential()
model.add(Embedding(vocab_size, 64, input_length = maxword))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
print(model.summary())



model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 20,batch_size = 100, verbose = 1)
score = model.evaluate(X_test, y_test)



print(score)







































































