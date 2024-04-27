import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(X_train,Y_train) , (X_test,Y_test)=mnist=tf.keras.datasets.mnist.load_data()

len(X_train)
X_train[0].shape
X_test[0].shape

plt.matshow(X_train[0])

#To improve the Accuracy
X_train=X_train/255
X_test=X_test/255

Y_train[:5]
len(Y_train)

#Converting into One Dimensional
X_train_Flattend=X_train.reshape(len(X_train),28*28)
X_test_Flattend=X_test.reshape(len(X_test),28*28)

X_train_Flattend[0]

#Neural Network Model
model=tf.keras.Sequential([
    tf.keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    tf.keras.layers.Dense(50,activation='relu'),
    tf.keras.layers.Dense(10,activation='sigmoid')])

model.compile(
 optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_Flattend,Y_train,epochs=5)

#Evaluation check is a good practice
model.evaluate(X_test_Flattend,Y_test)

y_predicted[0]

#Main Testing P+art **
plt.matshow(X_test[15])

np.argmax(y_predicted[15])

y_predicted_labels=[np.argmax(i) for i in y_predicted]
y_predicted_labels[:10]
#Matches or Not ? right : nop
Y_test[:10]

#checking prediction matrix
CM=tf.math.confusion_matrix(labels=Y_test,predictions=y_predicted_labels)

#Visualization Process
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(CM,annot=True,fmt='d')
plt.xlabel('Predicted Result')
plt.ylabel('Actual')