# %load_ext tensorboard
import datetime
from model import *
from data import *
# import torch
from tensorflow.python.client import device_lib


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# print ("---------- GPU torch --------------")
# print (torch.cuda.is_available())
# print (torch.cuda.device_count())
# print (torch.cuda.current_device())
print ("---------- GPU tensorflow --------------")
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print(device_lib.list_local_devices())

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.05,
                    zoom_range=0.05,
                    # brightness_range=(0.2,0.6),
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(16,r'data\train',r'data\label',data_gen_args,save_to_dir = None)
# myGene = trainGenerator(2,r'data\train\image',r'data\train\label',data_gen_args,save_to_dir = 'results')

model = unet()
model_checkpoint = ModelCheckpoint('unet_plastic.hdf5', monitor='loss',verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor="loss", min_delta=0.01, patience=2, verbose=1, mode="auto", baseline=None, restore_best_weights=True)
# model.fit(myGene,epochs=4,callbacks=[model_checkpoint, early_stopping])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(myGene,steps_per_epoch=64,epochs=50,callbacks=[model_checkpoint, tensorboard_callback])

testGene = testGenerator(r"data\test", num_image=50)
results = model.predict(testGene,50,verbose=1)
saveResult("data/test",results)
