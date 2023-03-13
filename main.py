from model import *
from data import *
from tensorflow.python.client import device_lib

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs:", len(physical_devices))
# print(device_lib.list_local_devices())

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.05,
                    zoom_range=0.05,
                    # brightness_range=(0.2,0.6),
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(16,r'data\train\image',r'data\train\label',data_gen_args,save_to_dir = 'results')
# myGene = trainGenerator(2,r'data\train\image',r'data\train\label',data_gen_args,save_to_dir = 'results')

model = unet()
model_checkpoint = ModelCheckpoint('unet_plastic.hdf5', monitor='loss',verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor="loss", min_delta=0.01, patience=2, verbose=1, mode="auto", baseline=None, restore_best_weights=True)
# model.fit(myGene,epochs=4,callbacks=[model_checkpoint, early_stopping])
model.fit(myGene,steps_per_epoch=1,epochs=4,callbacks=[model_checkpoint])

testGene = testGenerator(r"data\test", num_image=50)
results = model.predict(testGene,50,verbose=1)
saveResult("data/test",results)
