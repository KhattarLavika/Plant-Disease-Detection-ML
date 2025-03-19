# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 01:21:35 2024

@author: HP
"""

import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import filedialog,messagebox

#define class labels
class_labels=[
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato__healthy',
'Tomato___Bacterial_spot',
'Tomato__healthy',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites_Two-spotted_spider_mite']

#load resnet50 model

#model = models.resnet50(pretrained=True)
#fc=fully connected layer
#model.fc=torch.nn.Linear(2048,38)
#model.eval()

#data transformation
transformation=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_labels))  # Adjust for the number of classes
model.load_state_dict(torch.load('model.pth'))  # Load your trained model
model.eval()

#Loading the dataset
#dataset=ImageFolder(root='C:\\Users\\HP\\Desktop\\PlantDisease\\PlantVillage',transform=transformation)
#data_loader=DataLoader(dataset,batch_size=1,shuffle=True)

def predict_disease(image_path):
  image= Image.open(image_path)
  image=transformation(image).unsqueeze(0)
  with torch.no_grad():
    outputs=model(image)
  _, predicted=torch.max(outputs,1)
  return class_labels[predicted.item()]

def upload_image():
  file_path=filedialog.askopenfilename()
  image = Image.open(file_path)
  image = image.resize((200, 200))
# Convert the Image object into a Tkinter PhotoImage object
  photo = ImageTk.PhotoImage(image)
# Create a label to display the image
  selected_image.config(image=photo)
  selected_image.image = photo 
#add border to image
  selected_image.config(highlightthickness=1,highlightbackground='black')
  if file_path:
    try:
      disease_name=predict_disease(file_path)
      result.config(text=f'Predicted Disease: {disease_name}',font=('Arial',11,'bold'))
    except Exception as e:
      messagebox.showerror("Error",f"Error occured:{str(e)}")
     

root=tk.Tk()
root.title("Plant Disease Detection")
root.geometry("400x400")

upload_button=tk.Button(root,text="Upload Image",command=upload_image,font=('Arial',14,'bold'),bg='blue',fg='white')
upload_button.pack(pady=20)

result=tk.Label(root,highlightthickness=1,highlightbackground='black')
result.pack(pady=20)

selected_image = tk.Label(root)
selected_image.pack(pady=20)

root.mainloop()
