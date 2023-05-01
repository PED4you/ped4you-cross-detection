from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fastai.vision.all import *
from PIL import Image
import uvicorn

learn_inf = load_learner('models/model-2May.pkl')


def predict(img):
    pred, pred_idx, probs = learn_inf.predict(img)
    return pred, float(probs[pred_idx])


app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
    except Exception as e:
        return {"message": "There was an error uploading the file", 'exception': repr(e)}
    finally:
        file.file.close()

    try:
        # inference code here
        img = PILImage.create(contents)
        pred, prob = predict(img)
    except Exception as e:
        return {"message": "There was an error in inferencing", 'exception': repr(e)}

    return {"prediction": pred, "probability": prob}


if __name__ == "__main__":
    # read from command line args
    uvicorn.run(app, host='0.0.0.0', port=8000)
