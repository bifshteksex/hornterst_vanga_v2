from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import json
from typing import Dict
import os

app = FastAPI(title="DeepDanbooru API")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пути к файлам
MODEL_DIR = "model-resnet"
MODEL_PATH = os.path.join(MODEL_DIR, "model-resnet_custom_v3.h5")
TAGS_PATH = os.path.join(MODEL_DIR, "tags.txt")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.json")

# Загрузка модели
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy')  # добавьте эту строку

# Загрузка тегов
with open(TAGS_PATH, 'r') as f:
    tags = [line.strip() for line in f]

# Загрузка категорий (если нужно)
with open(CATEGORIES_PATH, 'r') as f:
    categories = json.load(f)

def process_image(image: Image.Image) -> np.ndarray:
    """Подготовка изображения для модели."""
    image = image.convert('RGB')
    image = image.resize((512, 512))
    image = np.array(image) / 255.
    return image

@app.post("/predict/", response_model=Dict[str, float])
async def predict(file: UploadFile = File(...)):
    """
    Получает изображение и возвращает предсказания тегов с вероятностями.
    """
    try:
        # Чтение и подготовка изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = process_image(image)
        
        # Получение предсказаний
        predictions = model.predict(np.expand_dims(image_array, axis=0))[0]
        
        # Формирование результата
        results = {
            tag: float(prob)  # преобразуем в float для корректной JSON сериализации
            for tag, prob in zip(tags, predictions)
            if prob > 0.5  # можно настроить порог вероятности
        }
        
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/tags/")
async def get_tags():
    """
    Возвращает список всех доступных тегов.
    """
    return {"tags": tags}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)