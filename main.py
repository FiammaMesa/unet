# %load_ext tensorboard
import datetime
from model import *
from data import *
from tensorflow.python.client import device_lib


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print ("---------- GPU tensorflow --------------")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
myGene = trainGenerator(16,'data/train','data/label',data_gen_args,save_to_dir = None)
# myGene = trainGenerator(2,r'data\train\image',r'data\train\label',data_gen_args,save_to_dir = 'results')

model = unet()
model_checkpoint = ModelCheckpoint('unet_plastic.hdf5', monitor='loss',verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor="loss", min_delta=0.01, patience=2, verbose=1, mode="auto", baseline=None, restore_best_weights=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# model.fit(myGene,steps_per_epoch=32,epochs=25,callbacks=[model_checkpoint, tensorboard_callback])
model.fit(myGene,steps_per_epoch=64,epochs=50,callbacks=[model_checkpoint, tensorboard_callback])

testGene = testGenerator(r"data/test", num_image=33)
results = model.predict(testGene,33,verbose=1)
saveResult("data/test",results)

data_test = "/content/drive/MyDrive/Colab Notebooks/unet/data/test/"
data_test_label = "/content/drive/MyDrive/Colab Notebooks/unet/data/test_label/"


calculatePixelsError(data_test, data_test_label)