import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import pickle

# === Load Class Mapping ===
with open("class_mapping.pkl", "rb") as f:
    class_mapping = pickle.load(f)
inv_class_mapping = {v: k for k, v in class_mapping.items()}

# === Load Trained Model ===
device = torch.device("cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_mapping))
model.load_state_dict(torch.load("cnn_autodrive_model.pth", map_location=device))
model = model.to(device)
model.eval()

# === Image Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Streamlit UI ===
st.title("Self-Driving Car Command Classifier")
st.write("Upload an image from the dashboard camera to predict the driving command.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and Predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        command = inv_class_mapping[predicted.item()]

    st.success(f"**Predicted Command:** {command}")
