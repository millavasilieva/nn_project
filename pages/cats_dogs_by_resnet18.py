import streamlit as st
import io

from PIL import Image
import requests

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import ToTensor,Compose,Resize,Normalize,CenterCrop


def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load('resnet18try.pt'))
    model.eval()
    return model


def load_image():
    uploaded_file = st.file_uploader(label='Выберите картинку формата jpg или jpeg')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def predict(model, image):
    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)


    class_ = int(torch.sigmoid(model(input_batch)).round().item())
    dog = 'Собака!'
    cat = 'Кошка!'
    if class_ == 1:
        st.write(f'Я думаю, что это... {dog}')
    if class_ == 0:
        st.write(f'Я думаю, что это... {cat}')
        

def main():
    st.title('Kлассификация кошек и собак')
    model = load_model()
    image = load_image()
    result = st.button('Нажмите для предсказания')
    if result:
        st.write('Обрабатываем результаты...')
        predict(model, image)
    

if __name__ == '__main__':
    main()