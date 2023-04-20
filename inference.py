from fastai.vision.all import *
from PIL import Image
from os import path

learn_inf = load_learner('models/model1.pkl')

# change below line to img destination #
basepath = './election-data/images/'
imgpath = 'bad-2023-04-14T13:24:21.545Z.png'

img = PILImage.create(path.join(basepath, imgpath))


# print(img)
pred, pred_idx, probs = learn_inf.predict(img)
print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
