from fastai.vision.all import *
from PIL import Image
from os import path

learn_inf = load_learner('models/model2.pkl')

# change below line to img destination #
basepath = './election-data/'
imgpath = 'input.png'


def predict(img):
    pred, pred_idx, probs = learn_inf.predict(img)
    return pred, float(probs[pred_idx])


img = PILImage.create(path.join(basepath, imgpath))
print(predict(img))
