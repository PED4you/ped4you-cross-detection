from fastapi import FastAPI, File, UploadFile

from fastai.vision.all import *
from PIL import Image

learn_inf = load_learner('models/model2.pkl')


def predict(img):
    pred, pred_idx, probs = learn_inf.predict(img)
    return pred, float(probs[pred_idx])


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        # inference code here
        img = PILImage.create(contents)
        pred, prob = predict(img)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"prediction": pred, "probability": prob}
