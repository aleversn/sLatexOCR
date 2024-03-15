import os
import sys
import json
import shutil
from fastapi import FastAPI, File, UploadFile, Header
from tempfile import NamedTemporaryFile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.body import response_body

from main.utils.jsonData import JsonData
from main.predictor import LatexOCR
from PIL import Image

with open('./app_config.json') as f:
    app_config = json.load(f)

config = JsonData('../config/colab.json')

model = LatexOCR(model_config=config, vocab_file='../model/tokenizer.json', resume_path=app_config['model_path'], resize_model_path=app_config['resize_model_path'])

app = FastAPI()

@app.get("/")
def home():
    res = response_body(message='SLatex OCR API')
    return res()

@app.post("/scan")
async def scan(api_key = Header(None), img: UploadFile = File(...)):
    valid = False
    if len(app_config['api_key']) == 0:
        valid = True
    for key in app_config['api_key']:
        if key == api_key:
            valid = True
            break
    if not valid:
        res = response_body(message='Invalid API Key', status=401)
        return res()
    save_dir = f"."
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    try:
        suffix = os.path.splitext(img.filename)[1]

        with NamedTemporaryFile(delete=True, suffix=suffix, dir=save_dir) as tmp:
            shutil.copyfileobj(img.file, tmp)
            print(os.path.join(save_dir, tmp.name))
            img_file = Image.open(os.path.join(save_dir, tmp.name))
            math = model(img_file)
    finally:
        img.file.close()
    
    res = response_body(message='File uploaded successfully', data=[math])
    return res()
