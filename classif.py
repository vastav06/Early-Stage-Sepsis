
import torch
from torchvision import transforms
from PIL import Image
import os
import torchvision

MODEL_WEB_PATHS = {
# base form of models trained on skin data
'HAM10000':'DDI-models/ham10000.pth',
'DeepDerm':'DDI-models/deepderm.pth',
'GroupDRO':'DDI-models/groupdro.pth',
'CORAL':   'DDI-models/coral.pth',
'CDANN':   'DDI-models/cdann.pth',
}

# thresholds determined by maximizing F1-score on the test split of the train 
#   dataset for the given algorithm
MODEL_THRESHOLDS = {
    'HAM10000':0.733,
    'DeepDerm':0.687,
    # robust training algorithms
    'GroupDRO':0.980,
    'CORAL':0.990,
    'CDANN':0.980,
}

def load_model(model_name, save_dir="DDI-models", download=True):
    """Load the model and download if necessary. Saves model to provided save 
    directory."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name.lower()}.pth")
    if not os.path.exists(model_path):
        if not download:
            raise Exception("Model not downloaded and download option not"\
                            " enabled.")
        else:
            # Requires installation of gdown (pip install gdown)
            import gdown
            gdown.download(MODEL_WEB_PATHS[model_name], model_path)

    # model = torchvision.models.inception_v3(pretrained=False, transform_input=True)
    model = torchvision.models.inception_v3(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2)
    model.AuxLogits.fc = torch.nn.Linear(768, 2)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model._ddi_name = model_name
    model._ddi_threshold = MODEL_THRESHOLDS[model_name]
    model._ddi_web_path = MODEL_WEB_PATHS[model_name]
    return model

def test_transform(image):
    """Apply the necessary transformations to the input image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjusted size for most models; change if needed for your specific model.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

def predict_image(image_path, model, threshold=0.5, use_gpu=False):
    image = Image.open(image_path).convert('RGB')
    
    transformed_image = test_transform(image)
    
    transformed_image = transformed_image.unsqueeze(0)
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    transformed_image = transformed_image.to(device)
    
    model.eval()
    
    # Predict
    with torch.no_grad():
        output = model(transformed_image)
    
    predicted_class_index = (output[:,1] >= threshold).long().item()
    class_names = ["benign", "malignant"]
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name


# image_path = "/Users/vastavbharambe/Desktop/abc/test_images/2.png"
# model_name = "CDANN"  # Adjust based on your specific use-case

# Assuming load_model is updated for current PyTorch/torchvision versions
# model = load_model(model_name, save_dir="DDI-Code-main/DDI-models", download=True)

# # Predict the class for the specified image
# predicted_class = predict_image(image_path, model, use_gpu=True)
# print(predicted_class)

