from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./skin_diseases_model1.h5")

# Class names for the different skin diseases
CLASS_NAMES = [
    "BA_cellulitis",
    "BA_impetigo",
    "FU_athlete_foot",
    "FU_nail_fungus",
    "FU_ringworm",
    "PA_cutaneous_larva_migrans",
    "VI_chickenpox",
    "VI_shingles"
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    # Resize the image to (224, 224)
    image = tf.image.resize(image, (224, 224))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
