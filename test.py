from fastai.vision.all import *
from PIL import Image
from os import path
import torch

basepath = './test/'
imgpath = 'bad-2023-04-20T14_04_02.311Z.png'

learn_inf = load_learner('models/model1.pkl').to('cpu')
img = torch.zeros(1, 3, 224, 224)
# img = PILImage.create(path.join(basepath, imgpath))
torch.onnx.export(learn_inf, img, 'onnx_model.onnx', verbose=True)