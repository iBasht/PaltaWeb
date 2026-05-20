import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import joblib
import torch.nn.functional as F
import datetime

# =====================================================================
# 1. CONFIGURACIÓN DEL DASHBOARD INDUSTRIAL (LAYOUT WIDE)
# =====================================================================
st.set_page_config(
    page_title="PaltoWeb - Visión IA", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- NUEVO: TEMA "APP MÓVIL" (Cabecera Oscura y Cuerpo Claro) ---
st.markdown("""
<style>
    /* Ocultar barra superior por defecto de Streamlit para usar la nuestra */
    [data-testid="stHeader"] { visibility: hidden; height: 0px; }

    /* Fondo general verde muy claro (como la app de ejemplo) */
    .stApp { background-color: #F0FDF4; }

    /* Botón gigante principal */
    .stButton>button {
        background-color: #15803D; /* Verde oscuro elegante */
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 20px;
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #166534; /* Más oscuro al pasar el cursor */
        transform: translateY(-2px);
    }

    /* Diseño del cuadro de subir imagen */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 2px dashed #86EFAC;
        border-radius: 15px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN TÉCNICA ---
device = torch.device("cpu")
class_names = ['Anthracnose', 'Healthy', 'Scab']
# Colores de recuadro para tema claro (pueden necesitar un ligero ajuste de saturación, pero los básicos están bien)
colors = {'Healthy': (0, 255, 0), 'Anthracnose': (0, 0, 255), 'Scab': (0, 255, 255)}

# =====================================================================
# 2. CARGAR TODO EL ECOSISTEMA DE IA (YOLO + 5 CLASIFICADORES)
# =====================================================================
@st.cache_resource
def cargar_ecosistema_ia():
    # --- 1. YOLO DETECTOR ---
    yolo_detector = YOLO('yolo26n.pt') 
    
    # --- RUTAS LOCALES DE PESOS ---
    path_res = 'modelo_resnet18_paltos.pth'
    path_dense = 'modelo_densenet_paltos.pth'
    path_eff = 'modelo_efficientnet_paltos.pth'
    path_rf = 'modelo_RF_VGG16.pkl'
    
    url_res = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_resnet18_paltos.pth"
    url_dense = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_DenseNet_paltos.pth"
    url_eff = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_EfficientNet_paltos.pth"
    url_rf = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_RF_VGG16.pkl"
    
    def descargar_cerebro(path, url, nombre_modelo):
        if not os.path.exists(path):
            with st.spinner(f"Sincronizando {nombre_modelo} con el servidor cloud..."):
                urllib.request.urlretrieve(url, path)

    descargar_cerebro(path_res, url_res, "ResNet18")
    descargar_cerebro(path_dense, url_dense, "DenseNet121")
    descargar_cerebro(path_eff, url_eff, "EfficientNetB2")
    descargar_cerebro(path_rf, url_rf, "VGG16+RF")
    
    # ResNet18
    res = models.resnet18(weights=None)
    res.fc = nn.Linear(res.fc.in_features, len(class_names))
    res.load_state_dict(torch.load(path_res, map_location=device))
    res.eval()
    
    # DenseNet121
    dense = models.densenet121(weights=None)
    dense.classifier = nn.Linear(dense.classifier.in_features, len(class_names))
    dense.load_state_dict(torch.load(path_dense, map_location=device))
    dense.eval()
    
    # EfficientNetB2
    eff = models.efficientnet_b2(weights=None)
    eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, len(class_names))
    eff.load_state_dict(torch.load(path_eff, map_location=device))
    eff.eval()
    
    # VGG16 + Random Forest
    vgg16_extractor = models.vgg16(weights='DEFAULT')
    vgg16_extractor.classifier = nn.Sequential(*list(vgg16_extractor.classifier.children())[:-1])
    vgg16_extractor = vgg16_extractor.to(device).eval()
    rf_decididor = joblib.load(path_rf)
    
    return yolo_detector, res, dense, eff, vgg16_extractor, rf_decididor

ecosistema = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================================
# 3. BARRA LATERAL (SIDEBAR)
# =====================================================================
with st.sidebar:
    st.title("⚙️ Configuración")
    st.write("**Usuario Activo:** Sebastian")
    st.write("**Empresa:** PaltosWeb - Visión IA")
    st.divider()
    conf_threshold = st.slider("Umbral de Confianza YOLO", min_value=0.10, max_value=1.00, value=0.50, step=0.05)
    st.caption("Ajusta la sensibilidad para la detección de los frutos en campo.")
    st.divider()
    if st.button("Cerrar Sesión", type="secondary"):
        st.info("Sesión finalizada.")

# =====================================================================
# 4. CUERPO PRINCIPAL DEL DASHBOARD: TABS Y PANELES
# =====================================================================
# =====================================================================
# 4. INTERFAZ UNIFICADA (ESTILO APP MÓVIL)
# =====================================================================

# 1. Cabecera Verde Oscura Superior
st.markdown("""
    <div style="background-color: #15803D; padding: 2.5rem 1rem 3rem 1rem; margin: -5rem -4rem 2rem -4rem; text-align: center; border-radius: 0 0 25px 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
        <h1 style="color: white; font-size: 3rem; margin: 0;">🥑 PaltoWeb</h1>
        <p style="color: #DCFCE7; font-size: 1.1rem; margin-top: 5px; font-weight: 500;">Detección Inteligente de Enfermedades</p>
    </div>
""", unsafe_allow_html=True)

# 2. Centramos el contenido simulando la pantalla de un móvil/tablet
col_izq, col_centro, col_der = st.columns([1, 2, 1])

with col_centro:
    
    # --- Registro de tiempo minimalista unificado ---
    ahora = datetime.datetime.now()
    st.markdown(f"<p style='text-align: center; color: #15803D; font-weight: bold; font-size: 1.1rem;'>📅 {ahora.strftime('%d / %m / %Y')} &nbsp;&nbsp;|&nbsp;&nbsp; 🕒 {ahora.strftime('%H:%M')}</p>", unsafe_allow_html=True)
    st.write("") # Espaciador
    
    # --- Captura Unificada ---
    archivo_subido = st.file_uploader("Sube una imagen para analizar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    contenedor_imagen = st.empty()
    
    if archivo_subido is not None:
        imagen_pil = Image.open(archivo_subido).convert('RGB')
        orig_img = np.array(imagen_pil)
        orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR) 
        
        contenedor_imagen.image(imagen_pil, use_container_width=True, style="border-radius: 15px;")
        
        # --- BOTÓN GIGANTE DE DETECCIÓN ---
        st.write("") # Espaciador
        if st.button("🔍 INICIAR DETECCIÓN DE ENFERMEDADES", use_container_width=True):
            with st.spinner("Analizando con Redes Neuronales..."):
                models_ia = cargar_ecosistema_ia()
                yolo_model = models_ia[0]
                res_model, dense_model, eff_model, vgg_model, rf_model = models_ia[1], models_ia[2], models_ia[3], models_ia[4], models_ia[5]
                
                resultados_yolo = yolo_model(orig_img_bgr, conf=conf_threshold)[0]
                boxes = resultados_yolo.boxes.xyxy.numpy().astype(int)
                
                if len(boxes) == 0:
                    st.warning("No se detectaron frutos en la imagen.")
                else:
                    detailed_results_list = []
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(orig_img_bgr.shape[1], box[2]), min(orig_img_bgr.shape[0], box[3])
                        recorte = orig_img_bgr[y1:y2, x1:x2]
                        recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
                        pil_recorte = Image.fromarray(recorte_rgb)
                        tensor = ecosistema(pil_recorte).unsqueeze(0)
                        fruit_id = f"P_{i+1:03d}"
                        
                        with torch.no_grad():
                            prob_res = F.softmax(res_model(tensor), dim=1).cpu().numpy()[0]
                            prob_dense = F.softmax(dense_model(tensor), dim=1).cpu().numpy()[0]
                            prob_eff = F.softmax(eff_model(tensor), dim=1).cpu().numpy()[0]
                            feat_vgg = vgg_model(tensor).cpu().numpy()
                            prob_rf = rf_model.predict_proba(feat_vgg)[0]
                            
                            prob_final = (prob_res + prob_dense + prob_eff + prob_rf) / 4.0
                            predicted_idx = np.argmax(prob_final)
                            pred_consensuado = class_names[predicted_idx]
                            confidence_score = prob_final[predicted_idx] * 100
                            
                        detailed_results_list.append({
                            "Fruto": fruit_id,
                            "Diagnóstico": pred_consensuado, 
                            "Porcentaje": confidence_score
                        })
                        
                        color = colors.get(pred_consensuado, (255, 255, 255))
                        cv2.rectangle(orig_img_bgr, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(orig_img_bgr, f"{pred_consensuado} {confidence_score:.1f}%", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    # Mostrar la imagen procesada
                    img_final_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
                    contenedor_imagen.image(img_final_rgb, caption="Análisis Completado", use_container_width=True)
                    
                    # --- TABLA DE RESULTADOS UNIFICADA ABAJO ---
                    st.divider()
                    st.markdown("<h3 style='color: #15803D;'>📊 Resultados de Enfermedades</h3>", unsafe_allow_html=True)
                    
                    df_resumen = pd.DataFrame(detailed_results_list)
                    st.dataframe(
                        df_resumen,
                        column_config={
                            "Fruto": "ID",
                            "Diagnóstico": st.column_config.TextColumn("Condición"),
                            "Porcentaje": st.column_config.ProgressColumn("Confianza", format="%.1f %%", min_value=0, max_value=100),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    sanas = sum(1 for r in detailed_results_list if r['Diagnóstico'] == 'Healthy')
                    enfermas = len(detailed_results_list) - sanas
                    st.info(f"**Resumen:** {len(boxes)} detectadas | 🟢 {sanas} Sanas | 🔴 {enfermas} Con afección")
