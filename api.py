from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

directory = "D:\Personal\Project\Plant Disease\Models\model_version_1"

MODEL = tf.keras.models.load_model(directory)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/")
async def root():
    return "Hello world"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
