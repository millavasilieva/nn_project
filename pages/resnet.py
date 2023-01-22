import streamlit as st

import wget
import os
import io
from PIL import Image


import torch
from torchvision.transforms import ToTensor,Compose,Resize,Normalize,CenterCrop


def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    return model

def load_labels():
    labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    labels_file = os.path.basename(labels_path)
    if not os.path.exists(labels_file):
        wget.download(labels_path)
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

def load_image():
    uploaded_file = st.file_uploader(label='Выберите картинку формата jpg или jpeg')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def predict(model, categories, image):
    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities,1)
    st.write("Я думаю, что это...")
    st.write(categories[top_catid[0]], top_prob[0].item())


def main():
    st.title('Классификация картинок')
    model = load_model()
    categories = load_labels()
    image = load_image()
    result = st.button('Нажми для предсказания')
    if result:
        st.write('Обрабатываем результаты...')
        predict(model, categories, image)


if __name__ == '__main__':
    main()