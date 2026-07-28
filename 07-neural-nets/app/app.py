import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import timm
import gradio as gr
from torchvision import transforms
from PIL import Image

try:
    import spaces
except ImportError:
    spaces = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(APP_DIR, 'xception_v4_23_val_acc_0.886.weights.h5')

classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants',
           'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']


class ClothingClassifier(nn.Module):
    def __init__(self, num_classes=10, inner_size=100, drop_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'legacy_xception',
            pretrained=True,
            num_classes=0,  # remove classifier
        )

        self.head = nn.Sequential(
            nn.Linear(2048, inner_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(inner_size, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def load_head_weights(model, weights_path):
    with h5py.File(weights_path, 'r') as f:
        head_layers = [
            ('head.0', ('dense_33',)),        # Linear(2048, 100)
            ('head.3', ('dense_34',)),        # Linear(100, 10)
        ]

        loaded = 0
        for layer_name, candidates in head_layers:
            linear = model.get_submodule(layer_name)

            for c in candidates:
                w_path = f'model_weights/{c}/{c}/kernel'
                b_path = f'model_weights/{c}/{c}/bias'
                if w_path not in f or b_path not in f:
                    continue

                kernel = torch.from_numpy(f[w_path][()].T)  # NH to HN
                bias = torch.from_numpy(f[b_path][()])

                if kernel.shape == linear.weight.shape:
                    linear.weight.data.copy_(kernel)
                    linear.bias.data.copy_(bias)
                    loaded += 1
                    break

        print(f'Loaded {loaded}/2 head layers')


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClothingClassifier()
load_head_weights(model, WEIGHTS_PATH)
model.to(device)
model.eval()


def predict_fn(image):
    if isinstance(image, dict):
        image = Image.fromarray(image['composite'])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)

    probs = torch.softmax(logits, dim=1)[0].cpu()
    top_idx = probs.argmax().item()
    return f"{classes[top_idx]}. probability is {probs[top_idx]:.0%}"


if spaces is not None:
    predict = spaces.GPU(predict_fn)
else:
    predict = predict_fn


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Textbox(label="Prediction"),
    title="Clothing Classifier",
    description="Upload a clothing image. Predicts: dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt",
)

iface.launch()
