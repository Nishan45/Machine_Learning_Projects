import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data=load_digits()
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target)

model=tf.keras.Sequential(
    [
        
        tf.keras.layers.Dense(units=100,activation='relu'),
       tf.keras.layers.Dense(units=80,activation='relu'),
       tf.keras.layers.Dense(units=60,activation='relu'),
       tf.keras.layers.Dense(units=40,activation='relu'),
       tf.keras.layers.Dense(units=25,activation='relu'),
       tf.keras.layers.Dense(units=15,activation='relu'),
       tf.keras.layers.Dense(units=10,activation='softmax')
        
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
model.fit(x_train,y_train,epochs=10)

pred=model.predict(x_test)

val=[]
for i in range(len(pred)):
    val.append(np.argmax(pred[i]))
    
c_matrix=confusion_matrix(y_test,val)
print(c_matrix)

report=classification_report(y_test,val)
print(report)
