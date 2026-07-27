import os
import gradio as gr
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(APP_DIR, 'xception_v4_23_val_acc_0.886.weights.h5')

base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)
base_model.trainable = False


def make_model(learning_rate=0.001, inner_size=100, drop_rate=0.2):
    inputs = keras.Input(shape=(299, 299, 3))
    base = base_model(inputs, training=False)
    pooling = keras.layers.GlobalAveragePooling2D()
    vectors = pooling(base)
    inner = keras.layers.Dense(inner_size, activation='relu')(vectors)
    dropout = keras.layers.Dropout(drop_rate)(inner)
    outputs = keras.layers.Dense(10)(dropout)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants',
           'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

model = make_model()
model(np.zeros((1, 299, 299, 3)))

with h5py.File(WEIGHTS_PATH, 'r') as f:
    def assign_if_match(weight, key_path):
        if weight.shape == f[key_path].shape:
            weight.assign(f[key_path][()])
            return True
        return False

    for layer in model.layers:
        if layer.name == 'xception':
            continue
        for w in layer.weights:
            wname = w.name.split('/')[-1].split(':')[0]
            # Search through the h5 file under model_weights/ for a matching tensor
            found = False
            for top_name in f['model_weights'].keys():
                if top_name in ('xception', 'top_level_model_weights', 'input_layer'):
                    continue
                candidate = f'model_weights/{top_name}/{top_name}/{wname}'
                if candidate in f and assign_if_match(w, candidate):
                    found = True
                    break
                candidate = f'model_weights/{top_name}/{wname}'
                if candidate in f and assign_if_match(w, candidate):
                    found = True
                    break
            if not found:
                print(f'WARNING: Could not load weights for {w.name}')


def predict(image):
    img = img_to_array(image.resize((299, 299)))
    X = np.array([img])
    X = preprocess_input(X)
    preds = model.predict(X, verbose=0)
    probs = tf.nn.softmax(preds[0]).numpy()
    top_idx = int(np.argmax(probs))
    return f"{classes[top_idx]}. probability is {probs[top_idx]:.0%}"


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Textbox(label="Prediction"),
    title="Clothing Classifier",
    description="Upload a clothing image. Predicts: dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt"
)

iface.launch()
