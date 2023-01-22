import streamlit as st
import io

from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor,Compose,Resize,Normalize,CenterCrop


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(800, 256)
        self.do      = nn.Dropout()
        self.linear2 = nn.Linear(256, 4)


    def forward(self, x):

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        flat_out = self.flat(layer3_out)
        
        linear1_out = torch.relu(self.do(self.linear1(flat_out)))
        linear2_out = torch.relu(self.linear2(linear1_out))

        return linear2_out

def load_model():
    model = CNN()
    model = model.load_state_dict(torch.load('bloodnew.pt'))
    
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
    # model.eval()
    # with torch.no_grad():
    #     output = model(input_batch)

    # probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # top_prob, top_catid = torch.topk(probabilities,1)
    st.write("Я думаю, что это...")
    class_ = model(input_tensor.unsqueeze(0)).argmax().item()
    # predictions = model(input_batch)
    # class_ = F.softmax(predictions[0], dim=0)
    # classes = ['EOCINOPHIL', 'LYMPHOCYTE','MONOCYTE', 'NEUTROPHIL']
    
    # st.write(class_)
    one = 'EOCINOPHIL'
    two = 'LYMPHOCYTE'
    three = 'MONOCYTE'
    four = 'NEUTROPHIL'
    st.write(class_)
    # if class_ == 0:
    #     st.write(f'Predicted class: {one}')
    # if class_ == 1:
    #     st.write(f'Predicted class: {two}')
    # if class_ == 2:
    #     st.write(f'Predicted class: {three}')
    # if class_ == 3:
    #     st.write(f'Predicted class: {four}')  
    

def main():
    st.title('Классификация клеток крови')
    model = load_model()
    # categories = load_labels()
    image = load_image()
    result = st.button('Нажми для предсказания')
    if result:
        st.write('Обрабатываем результаты...')
        predict(model, image)


if __name__ == '__main__':
    main()