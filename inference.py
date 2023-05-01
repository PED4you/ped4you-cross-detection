from fastai.vision.all import *
from pathlib import Path

learn_inf = load_learner('models/model2.pkl')



def predict(img):
    pred, pred_idx, probs = learn_inf.predict(img)
    return pred, float(probs[pred_idx])



if __name__ == "__main__":
    # change below line to img destination #
    basepath = Path('.')
    imgpath = 'input.png'

    img = PILImage.create(basepath/ imgpath)

    print(predict(img))
