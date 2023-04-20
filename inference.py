from fastai.vision.all import *
from PIL import Image

learn_inf = load_learner('models/model1.pkl')
# change below line to img destination #
img = PILImage.create('/Users/vikimark/Documents/PythonFlow/Ped4You-CrossRecognition/dataset/positive/good-2023-04-14T08:52:54.453Z.png')
# print(img)
pred,pred_idx,probs = learn_inf.predict(img)
print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')