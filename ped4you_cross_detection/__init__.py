from fastai.vision.all import load_learner

def predictOneTime(img):
    learn_inf = load_learner('models/model-2May-15th.pkl')
    pred, pred_idx, probs = learn_inf.predict(img)
    return pred, float(probs[pred_idx])

