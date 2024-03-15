# Simplify Latex OCR

### Quick Start Training and Evaluation

**Install**

```bash
pip install -r requirements.txt
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