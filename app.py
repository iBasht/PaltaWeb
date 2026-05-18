import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# --- DISEÑO DE LA PÁGINA ---
st.set_page_config(page_title="Diagnóstico de Paltas", layout="centered")
st.title("🥑 IA Agrícola: Diagnóstico de Paltas")
st.write("Sube una foto tomada en campo para detectar y analizar el estado de los frutos.")

# --- CONFIGURACIÓN ---
device = torch.device("cpu") # En web usamos CPU
class_names = ['Anthracnose', 'Healthy', 'Scab']
colors = {'Healthy': (0, 255, 0), 'Anthracnose': (0, 0, 255), 'Scab': (0, 255, 255)}

# --- CARGAR MODELOS (En Caché para que cargue rápido) ---
@st.cache_resource
def cargar_ia():
    # Cargar YOLO
    yolo = YOLO('yolo26n.pt') 
    
    # Cargar ResNet18
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 3)
    resnet.load_state_dict(torch.load('modelo_resnet18_paltos.pth', map_location=device))
    resnet.eval()
    
    return yolo, resnet

transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- INTERFAZ DE USUARIO ---
archivo_subido = st.file_uploader("Arrastra aquí tu foto (.jpg, .png)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    orig_img = np.array(imagen_pil)
    orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR) 
    
    st.image(imagen_pil, caption="Foto Original", use_container_width=True)
    
    if st.button("Ejecutar Diagnóstico", type="primary"):
        with st.spinner("Las redes neuronales están analizando..."):
            yolo_model, resnet_model = cargar_ia()
            
            # 1. Cazador YOLO
            resultados_yolo = yolo_model(orig_img_bgr, conf=0.70)[0]
            boxes = resultados_yolo.boxes.xyxy.numpy().astype(int)
            
            if len(boxes) == 0:
                st.warning("YOLO no encontró ninguna palta clara en esta imagen.")
            else:
                conteo = {name: 0 for name in class_names}
                
                # 2. Doctor ResNet
                for box in boxes:
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(orig_img_bgr.shape[1], box[2]), min(orig_img_bgr.shape[0], box[3])
                    recorte = orig_img_bgr[y1:y2, x1:x2]
                    recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
                    pil_recorte = Image.fromarray(recorte_rgb)
                    
                    tensor = transformaciones(pil_recorte).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = resnet_model(tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        max_prob, idx = torch.max(probs, 0)
                    
                    clase = class_names[idx.item()]
                    confianza = max_prob.item() * 100
                    conteo[clase] += 1
                    
                    # Dibujar
                    color = colors.get(clase, (255, 255, 255))
                    cv2.rectangle(orig_img_bgr, (x1, y1), (x2, y2), color, 3)
                    etiqueta = f"{clase} {confianza:.1f}%"
                    cv2.putText(orig_img_bgr, etiqueta, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                
                # Mostrar Resultado
                img_final_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_final_rgb, caption="Resultado del Análisis", use_container_width=True)
                
                st.success(f"¡Análisis exitoso! {len(boxes)} paltas procesadas.")
                st.json(conteo) # Muestra un pequeño resumen estructurado
