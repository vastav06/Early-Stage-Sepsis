import classif
import area
image_path = "/Users/vastavbharambe/Downloads/cap/realtimeimages/h3nw.jpeg"

json_file_path = "/Users/vastavbharambe/Downloads/cap/jsonfile/area_data.json"
model_name = "CDANN"  # Adjust based on your specific use-case

# Assuming load_model is updated for current PyTorch/torchvision versions
model = classif.load_model(model_name, save_dir="DDI-Code-main/DDI-models", download=True)

# Predict the class for the specified image

predicted_class = classif.predict_image(image_path, model, use_gpu=True)
print(predicted_class)
if predicted_class == "benign" or predicted_class == "malignant":
    
    area.process_image(image_path, json_file_path)
    
else:
    print("There is no detection of early stage of sepsis.")
