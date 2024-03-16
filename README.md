# Simplify Latex OCR

### Quick Start Training and Evaluation

This repo is an open resource image-to-latex project based on `pix2tex`.

**Install**

```bash
pip install -r requirements.txt
```

**Download Model**

Image Resizer Model

```bash
wget https://drive.google.com/file/d/1U1raFeRG2LUCSkF0ozu2ByOOmu3lAJHF/view?usp=drive_link
```

Latex OCR Model

```bash
wget https://drive.google.com/file/d/1dTVdfl7CU8YQxRxhBwfm8lwgH-IQovxi/view?usp=drive_link
```

**Run Code**

See `playground.ipynb`

Training

```python
from main.utils.jsonData import JsonData
from main.trainer import Trainer

config = JsonData('./config/colab.json')

trainer = Trainer(model_config=config, vocab_file='./model/tokenizer.json', train_data_path=[...], eval_data_path=[...], resume_path='...', batchsize=30, pad=True, task_name='...')

for i in trainer(num_epochs=120, lr=1e-4):
    a = i
```

Pred

```python
from main.utils.jsonData import JsonData
from main.predictor import LatexOCR

config = JsonData('./config/colab.json')

model = LatexOCR(model_config=config, vocab_file='./model/tokenizer.json', resume_path='./save_model/released/Checkpoint_8146/model.pth', resize_model_path='./model/image_resizer.pth')

import os
from PIL import Image

dir = './test_samples'
imgs = ['1.png', 'a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png', 'j.png']

for path in imgs:
    img = Image.open(os.path.join(dir, path))
    math = model(img)
    print(f'### {path}\n${math}$\n')
```

### API

**Make Config**

```bash
cp config/app_config.json.example api/app_config.json
```

You can set you own `api_key` in `api/app_config.json`, otherwise, the app will run without authentication.

**Start Server**

```bash
cd api/
```

```bash
uvicorn app:app --host=0.0.0.0 --port=8000
```

### Docker

Build and run the docker image

```bash
docker compose up -d --build
```

The code is deploy based on shared storage, so that you can modify the code in the physical machine and the docker container will automatically update the code. So, do not delete any necessary code files in the physical machine.

Restart the docker container

```bash
docker compose restart
```

### License
MIT License

Copyright (c) 2024 Creator SN®

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.