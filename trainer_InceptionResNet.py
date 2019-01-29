from keras import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Sequential
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras import backend as K
import os
#from sklearn.metrics import classification_report
import time
from pprint import pprint
#from PIL import Image

class EndCallback(Callback):
    def __init__(self, timestr):
        self.timestr = timestr   

    def on_epoch_end(self, epoch, logs={}):
        with open('logs/'+timestr+'_trainlog.txt', 'a') as out:
            pprint('Epoch: '+str(epoch), stream=out)
            pprint(logs, stream=out)
            
class LrReducer(Callback):
    def __init__(self, patience=1, reduce_rate=0.5, reduce_nb=6, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = K.get_value(self.model.optimizer.lr)
                    #lr = self.model.optimizer.lr.get_value()
                    K.set_value(self.model.optimizer.lr,lr*self.reduce_rate)
                    new_lr = K.get_value(self.model.optimizer.lr)
                    #self.model.optimizer.lr.set_value(lr*self.reduce_rate)
                    if self.verbose > 0:
                        print('---LR reduced to: %f' % new_lr)
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1
                
def fileList(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith('.jpg'):
                matches.append(os.path.join(root, filename))
    return len(matches)

train_dir = TRAIN_DIR
val_dir = VAL_DIR
base_dir = BASE_DIR

#parameters
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001
DECAY = 0.001
MOMENTUM = 0.9
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
timestr = time.strftime("%Y%m%d-%H%M%S")
NAME = f"{EPOCHS}-EPOCHS-{str(LR)[2:]}-LR-{timestr}"

#savepath
model_path = f'models/{timestr}'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# dimensions of our images.
img_size = (300, 300)
in_shape = (300,300,3)
input = Input(shape=in_shape,name = 'image_input')

#model
base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling='avg')

top_model = Sequential()
top_model.add(Dense(1, input_shape=(base_model.output_shape[1:]), activation='sigmoid'))
model = Model(inputs=base_model.input, outputs= top_model(base_model.output))

adam = Adam(lr=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

model.summary()

with open('logs/'+timestr+'_trainlog.txt', 'wt') as out:
    model.summary(print_fn=lambda x: out.write(x + '\n'))

train_size = fileList(train_dir)
val_size = fileList(val_dir)

#prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    rotation_range=90,
    horizontal_flip=True,
    samplewise_center=False,
    samplewise_std_normalization=False,
    preprocessing_function=preprocess_input
    )

val_datagen = ImageDataGenerator(samplewise_center=False,
    samplewise_std_normalization=False,
    preprocessing_function=preprocess_input
    )

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    class_mode='binary')

logCallback = EndCallback(timestr)
lr_reduce = LrReducer()

lr_current = K.get_value(model.optimizer.lr)
filepath = model_path+"/ResNet-{epoch:02d}-{val_acc:.3f}.h5"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones

model.fit_generator(
    train_generator,
    steps_per_epoch=int(train_size/BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=int(val_size/BATCH_SIZE), callbacks=[logCallback, lr_reduce, checkpoint])

model.save_weights("models/{}_weights.h5".format(NAME))
model.save("models/{}.h5".format(NAME))