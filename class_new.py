import h5py
import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model 
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image




IMAGE_SIZE = [224, 224]                                                              #specifying the image size
train_path = 'C:/Users/divya/Desktop/fake_note_project/data/training'                #assigning the training set path
valid_path = 'C:/Users/divya/Desktop/fake_note_project/data/test'                    #assigning the testing set path





vgg = VGG16(input_shape=IMAGE_SIZE + [3],weights='imagenet', include_top=False,classes=7)  #vgg1

for layer in vgg.layers:
  layer.trainable = False

folders = glob('C:/Users/divya/Desktop/fake_note_project/data/training/*')            #to include the folders in training set

x = Flatten()(vgg.output)                                                             #flatenning the layer

prediction = Dense(len(folders),activation='softmax')(x)                              

model = Model(inputs=vgg.input,outputs=prediction)                                    #creating a model

#model.summary()

model.compile(                                                                        #compiling the model
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics = ['accuracy']
) 






from keras.preprocessing.image import ImageDataGenerator                             #creating objects for ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('C:/Users/divya/Desktop/fake_note_project/data/training',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 classes=['10','100','20','200','2000','50','500'],
                                                 class_mode='categorical')
test_set = train_datagen.flow_from_directory('C:/Users/divya/Desktop/fake_note_project/data/test',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 classes=['10','100','20','200','2000','50','500'],
                                                 class_mode='categorical'
                                                 )



# In[4]:


checkpoint_path="C:/Users/divya/Desktop/fake_note_project/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)


# In[5]:


def classifying():
   r = model.fit(                                                                    #training the model   
       training_set,
        validation_data=test_set,
        epochs=15,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
        callbacks = [cp_callback])
   plt.plot(r.history['loss'],label='train loss')                                    
   plt.plot(r.history['val_loss'],label='val loss')
   plt.legend()
   plt.show()
   plt.savefig('Lossval_loss')

   plt.plot(r.history['accuracy'],label='train acc')
   plt.plot(r.history['val_accuracy'],label='val acc')
   plt.legend()
   plt.show()
   plt.savefig('AccVal_acc')
#classifying()


# In[6]:


latest=tf.train.latest_checkpoint(checkpoint_dir)
latest


# In[7]:


from keras.preprocessing import image
train_generator=train_datagen.flow_from_directory(
        directory=r"C:/Users/divya/Desktop/fake_note_project/data/training",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )

test_generator=test_datagen.flow_from_directory(
        directory=r"C:/Users/divya/Desktop/fake_note_project/data/test",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )
model.load_weights(latest)


# In[23]:


#file_path='C:/Users/divya/Desktop/fake_note_project/data/validation/500/500_12.jpg'
def pred(file_path):
    img1 = image.load_img(file_path,target_size=(224,224))                    #load the image
    img1 = np.asarray(img1)                                                   #convert into nparray
    img1 = np.expand_dims(img1, axis=0)                                       #expand the dimensions
    res=model.predict(img1)                                                   #predict the class
    predicted_class_indices=np.argmax(res,axis=1)
    labels=(train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions=[labels[k] for k in predicted_class_indices]
    print(predictions)
    return predictions


# In[ ]:




