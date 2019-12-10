from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import SaveModelCallback
import os

np.random.seed(2)

data = ImageDataBunch.from_folder('Processed_data',train='train', test = 'test',valid_pct = 0.2,  ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.), size=128, bs=32).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet32, metrics=error_rate)

learn.fit_one_cycle(6, callbacks=[SaveModelCallback(learn)])

learn.save('stage-1_v1_32')

tests = os.listdir('Processed_data/test')

for i in range(len(tests)):
	tests[i] = tests.split('.')[0]

df_test = [['id','concrete_cement','healthy_metal','incomplete','irregular_metal','other']]
for i in tests:
  img = open_image('Processed_data/test/'+i+'.tif')
  ar = np.array(learn.predict(img)[2])
  temp = [i]
  ar = temp.append(ar)
  temp.append(ar)

df_test = pd.DataFrame(df_test)
df_test.to_csv('submission.csv')





