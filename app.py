import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import yaml
import os
import google.generativeai as genai  # Gemini AI integration

# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyAk-ZcH4SjtYlGklCIZ6lpizZXW-BaKNcc")

# Load class names from data.yaml
def load_class_names(yaml_path):
    if not os.path.exists(yaml_path):
        st.error("data.yaml file not found!")
        return []
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('names', [])

# ✅ Load YOLOv5 model correctly
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', autoshape=False)
    model.eval()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    model = None

# Set YAML file path
model_folder = os.path.dirname('best.pt')
yaml_file = os.path.join(model_folder, 'data.yaml')
class_names = load_class_names(yaml_file)

# ✅ Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# ✅ Perform classification with YOLO
def classify_image(image):
    if model is None:
        return None, 0.0
    image_tensor = preprocess_image(image)
    results = model(image_tensor)
    probabilities = results.softmax(1)
    pred_class_idx = probabilities.argmax(1).item()
    confidence = probabilities.max().item()
    pred_class = class_names[pred_class_idx] if pred_class_idx < len(class_names) else "Unknown"
    return pred_class, confidence

# ✅ Get disease details using Gemini AI (FIXED)
def get_disease_details(pred_class, lang="en"):
    prompts = {
        "en": f"Provide detailed information on {pred_class} disease in sugarcane. Include:\n"
              f"1. How this disease affects sugarcane.\n"
              f"2. Weather conditions that contribute to the disease.\n"
              f"3. Soil conditions that may cause the disease.",
        
        "ta": f"கரும்பு செடிகளில் {pred_class} நோய் பற்றிய விரிவான தகவலை வழங்கவும். இதில் அடங்கும்:\n"
              f"1. இந்த நோய் கரும்பை எவ்வாறு பாதிக்கிறது.\n"
              f"2. எந்த வானிலை நிலைமைகள் இந்த நோயை அதிகரிக்கச் செய்யும்.\n"
              f"3. எந்த மண் சூழ்நிலைகள் இந்த நோய்க்கு காரணமாக இருக்கலாம்."
    }

    try:
        # ✅ Use Gemini AI's generate_content correctly
        model_gemini = genai.GenerativeModel("gemini-pro")  # Initialize the correct model
        response = model_gemini.generate_content(prompts[lang])  
        return response.text  # Extract the text response
    except Exception as e:
        return f"Error retrieving information: {e}"

# ✅ Provide recommendations
def provide_recommendations(pred_class, lang="en"):
    recommendations = {
    'Rust': {
        'en': '''> Apply fungicides like Mancozeb or Triazole-based compounds at the early onset of symptoms.
                  > Use resistant sugarcane varieties where possible.
                  > Maintain proper plant spacing and ventilation to reduce humidity, which can reduce the severity of the disease.
                  > Remove infected leaves and improve soil health to support the plants’ immune responses.''',
        'ta': '''> தொடக்கத்திலேயே Mancozeb அல்லது Triazole போன்ற பூச்சிக்கொல்லிகளை பயன்படுத்தவும்.
                  > நோய் எதிர்ப்பு கரும்பு வகைகளை பயிரிடவும்.
                  > ஈரப்பதத்தை குறைக்க தடுப்பு இடைவெளி மற்றும் காற்றோட்டத்தை பராமரிக்கவும்.
                  > பாதிக்கப்பட்ட இலைகளை அகற்றவும் மற்றும் நிலத்தின் ஆரோக்கியத்தை மேம்படுத்தவும்.'''
    },
    'RedRot': {
        'en': '''> Select disease-resistant sugarcane varieties for planting.
                  > Apply fungicides like Carbendazim at the first sign of disease.
                  > Improve drainage in the field to prevent waterlogging, which exacerbates red rot.
                  > Rotate crops and practice field sanitation to prevent the pathogen from persisting in the soil.''',
        'ta': '''> நோய் எதிர்ப்பு கரும்பு வகைகளை தேர்ந்தெடுத்து பயிரிடவும்.
                  > நோய் தொடங்கிய உடனேயே Carbendazim போன்ற பூச்சிக்கொல்லிகளை பயன்படுத்தவும்.
                  > வயலில் நீர் தேக்கம் ஏற்படாமல் வடிகால் வசதியை மேம்படுத்தவும்.
                  > நெறிகளை மாற்றி பயிரிடுதல் மற்றும் வயல் சுகாதாரத்தை கடைப்பிடிக்கவும்.'''
    },
    'Mosaic': {
        'en': '''> Use resistant sugarcane varieties to prevent the spread of the disease.
                  > Control aphids and other vectors that transmit the mosaic virus.
                  > Remove and destroy infected plants to limit further spread.
                  > Ensure proper field hygiene to avoid contamination between fields.''',
        'ta': '''> நோயின் பரவலைத் தடுக்க நோய் எதிர்ப்பு கரும்பு வகைகளை பயன்படுத்தவும்.
                  > மோசைக் வைரஸை பரப்பும் நரம்புத் தொந்திகள் மற்றும் பிற பக்கவாதிகளை கட்டுப்படுத்தவும்.
                  > பாதிக்கப்பட்ட செடிகளை அகற்றி அழிக்கவும்.
                  > நிலங்களுக்கிடையே தொற்றுநோய் பரவாமல் தடுக்க வயல் சுகாதாரத்தை உறுதி செய்யவும்.'''
    },
    'Yellow': {
        'en': '''> Remove and destroy affected plants to avoid spreading the virus.
                  > Use certified disease-free planting material.
                  > Apply nitrogen fertilizers to boost plant growth and immunity.
                  > Manage aphids and other insect vectors that may spread the disease.''',
        'ta': '''> வைரஸின் பரவலைத் தடுக்க பாதிக்கப்பட்ட செடிகளை அகற்றி அழிக்கவும்.
                  > சான்று பெற்ற நோயற்ற விதை பொருட்களை பயன்படுத்தவும்.
                  > செடிகளின் வளர்ச்சி மற்றும் நோய் எதிர்ப்பை மேம்படுத்த நைட்ரஜன் உரங்களைப் பயன்படுத்தவும்.
                  > நோயை பரப்பக்கூடிய நரம்புத் தொந்திகள் மற்றும் பிற பூச்சிகளை நிர்வகிக்கவும்.'''
    },
    'healthy': {
        'en': "> No action needed, the plant is healthy.",
        'ta': "> எந்த நடவடிக்கையும் தேவையில்லை, செடி ஆரோக்கியமாக உள்ளது."
    }
}
    return recommendations.get(pred_class, {}).get(lang, "No specific recommendation available.")

# ✅ Disease name translations
disease_labels = {
    'Rust': {'en': 'Rust', 'ta': 'இலைக்கறை நோய்'},
    'RedRot': {'en': 'RedRot', 'ta': 'சிவப்பு அழுகல்'},
    'Mosaic': {'en': 'Mosaic', 'ta': 'மொசைக்/தேமல்நோய்'},
    'Yellow': {'en': 'Yellow', 'ta': 'மஞ்சள் நோய்'},
    'healthy': {'en': 'Healthy', 'ta': 'ஆரோக்கியமான'}
}

# ✅ Translations
translations = {
    "title": {"en": "Sugarcane Leaf Disease Prediction & Recommendations", "ta": "கரும்பு இலை நோய் கணிப்பு"},
    "upload_label": {"en": "Choose an image...", "ta": "ஒரு படத்தைத் தேர்ந்தெடுக்கவும்..."},
    "uploaded_caption": {"en": "Uploaded Image", "ta": "பதிவேற்றப்பட்ட படம்"},
    "predicted_class": {"en": "Predicted Disease", "ta": "முன்கணிக்கப்பட்ட நோய்"},
    "disease_info": {"en": "Disease Information", "ta": "நோய் பற்றிய தகவல்"},
    "recommendation": {"en": "Recommendations", "ta": "பரிந்துரைகள்"},
}

# ✅ Streamlit App
language = st.radio("Choose Language / மொழி தேர்வு:", ["English", "தமிழ்"])
lang_code = "ta" if language == "தமிழ்" else "en"

st.title(translations["title"][lang_code])

# ✅ File Uploader
uploaded_file = st.file_uploader(translations["upload_label"][lang_code], type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=translations["uploaded_caption"][lang_code], use_column_width=False, width=300)
    pred_class, confidence = classify_image(image)

    if pred_class:
        disease_name = disease_labels.get(pred_class, {}).get(lang_code, pred_class)
        recommendation = provide_recommendations(pred_class, lang=lang_code)

        # ✅ Fetch disease details from Gemini AI
        disease_info = get_disease_details(pred_class, lang=lang_code)

        # ✅ Display results
        st.write(f"**{translations['predicted_class'][lang_code]}:** {disease_name}")
        st.write(f"**{translations['disease_info'][lang_code]}:**")
        st.write(disease_info)
        st.write(f"**{translations['recommendation'][lang_code]}:** {recommendation}")
    else:
        st.error("Model is not loaded. Unable to classify.")
