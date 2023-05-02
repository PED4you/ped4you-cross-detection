from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fastai.vision.all import *
from PIL import Image
import uvicorn

learn_inf = load_learner('models/model-2May-14th.pkl')
# case of two cross & 3rd, 4th of bad cross
# case of 2nd, 5th good cross


def predict(img):
    pred, pred_idx, probs = learn_inf.predict(img)
    return pred, float(probs[pred_idx])


app = FastAPI(
    title="ped4you-cross-detection",
    version="2May-14th",
    docs_url='/'
)

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def check_health():
    return {"status": "ok"}

@app.get("/warmup")
async def warm_up():
    return {"status": "ok"}

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
