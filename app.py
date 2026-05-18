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

# =====================================================================
# 1. CONFIGURACIÓN DE PÁGINA ANCHA (DASHBOARD INDUSTRIAL)
# =====================================================================
st.set_page_config(
    page_title="SAGIE Agro - Visión IA", 
    layout="wide", # ¡Esto activa el modo pantalla completa!
    initial_sidebar_state="expanded"
)

# --- CONFIGURACIÓN DE MODELOS ---
device = torch.device("cpu")
class_names = ['Anthracnose', 'Healthy', 'Scab']
colors = {'Healthy': (0, 255, 0), 'Anthracnose': (0, 0, 255), 'Scab': (0, 255, 255)}

@st.cache_resource
def cargar_ia():
    yolo = YOLO('yolo26n.pt') 
    resnet_path = 'modelo_resnet18_paltos.pth'
    
    if not os.path.exists(resnet_path):
        # Aquí se mantiene tu enlace automático de GitHub Releases
        url_modelo = "TU_LINK_DE_GITHUB_RELEASES_AQUÍ" 
        urllib.request.urlretrieve(url_modelo, resnet_path)

    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 3)
    resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet.eval()
    return yolo, resnet

transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================================
# 2. BARRA LATERAL (SIDEBAR) - CONTROL DE USUARIO Y CONFIGURACIÓN
# =====================================================================
with st.sidebar:
    st.title("⚙️ Configuración")
    st.write("**Usuario Activo:** Sebastian")
    st.write("**Sistema:** SAGIE Agro v1.0")
    st.divider()
    
    # Control deslizante para el umbral de YOLO (Pedido en tu boceto)
    conf_threshold = st.slider("Umbral de Confianza YOLO", min_value=0.10, max_value=1.00, value=0.70, step=0.05)
    st.caption("Ajusta la sensibilidad para la detección de los frutos en campo.")
    st.divider()
    if st.button("Cerrar Sesión", type="secondary"):
        st.info("Sesión finalizada.")

# =====================================================================
# 3. BARRA SUPERIOR (NAVBAR SIMULADO CON TABS)
# =====================================================================
st.title("🥑 SAGIE Agro - Visión IA")
pestana_monitoreo, pestana_historial, pestana_reportes = st.tabs([
    "📺 Monitoreo en Vivo", 
    "📜 Historial de Análisis", 
    "📊 Reportes y Rendimiento"
])

# --- CONTENIDO DE LA PESTAÑA PRINCIPAL ---
with pestana_monitoreo:
    
    # Creamos la distribución de 3 columnas de tu dibujo (Izquierda, Centro, Derecha)
    col_izquierda, col_central, col_derecha = st.columns([1, 2, 1])
    
    # -----------------------------------------------------------------
    # PANEL IZQUIERDO: KPI y Monitoreo Individual
    # -----------------------------------------------------------------
    with col_izquierda:
        st.subheader("📋 Estado de Lote")
        st.metric(label="Lote Actual", value="LOTE-2026A", delta="Fundo Trujillo")
        
        st.write("**Monitoreo Individual:**")
        # Tabla dinámica simulada para la barra lateral izquierda
        data_monitoreo = {
            "ID Fruto": ["Palta_042", "Palta_043", "Palta_044"],
            "Diagnóstico": ["🔴 Anthracnose", "🟢 Healthy", "🟡 Scab"]
        }
        st.table(pd.DataFrame(data_monitoreo))

    # -----------------------------------------------------------------
    # PANEL CENTRAL: Visualizador YOLO (Núcleo Visual)
    # -----------------------------------------------------------------
    with col_central:
        st.subheader("🖼️ Visualizador YOLOv8 + ResNet18")
        archivo_subido = st.file_uploader("Cargar captura de inspección", type=["jpg", "jpeg", "png"])
        
        # Marcadores de posición para las imágenes
        contenedor_imagen = st.empty()
        
        if archivo_subido is not None:
            imagen_pil = Image.open(archivo_subido).convert('RGB')
            orig_img = np.array(imagen_pil)
            orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR) 
            
            contenedor_imagen.image(imagen_pil, caption="Imagen cargada en el visor", use_container_width=True)
            
            if st.button("🚀 INICIAR ESCANEO DE ALTA RESOLUCIÓN", type="primary", use_container_width=True):
                with st.spinner("Ejecutando Pipeline de Redes Neuronales..."):
                    yolo_model, resnet_model = cargar_ia()
                    
                    # 1. YOLO con el umbral dinámico del slider
                    resultados_yolo = yolo_model(orig_img_bgr, conf=conf_threshold)[0]
                    boxes = resultados_yolo.boxes.xyxy.numpy().astype(int)
                    
                    if len(boxes) == 0:
                        st.warning("No se detectaron frutos con el umbral actual.")
                    else:
                        conteo = {name: 0 for name in class_names}
                        
                        # 2. Iteración y clasificación
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
                            
                            color = colors.get(clase, (255, 255, 255))
                            cv2.rectangle(orig_img_bgr, (x1, y1), (x2, y2), color, 3)
                            etiqueta = f"{clase} {confianza:.1f}%"
                            cv2.putText(orig_img_bgr, etiqueta, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        
                        # Reemplazar imagen original con la procesada por la IA
                        img_final_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
                        contenedor_imagen.image(img_final_rgb, caption="Diagnóstico en tiempo real completado", use_container_width=True)
                        
                        # Guardar en el estado de Streamlit para usarlo en los otros paneles
                        st.session_state['conteo_actual'] = conteo
                        st.session_state['total_paltas'] = len(boxes)
                        st.success("Inspección finalizada con éxito.")

    # -----------------------------------------------------------------
    # PANEL DERECHO: Métricas Globales del Lote
    # -----------------------------------------------------------------
    with col_derecha:
        st.subheader("📈 Resumen Ejecutivo")
        
        # Verificamos si ya se corrió el modelo para mostrar datos reales, si no, mostramos ceros
        total_p = st.session_state.get('total_paltas', 0)
        conteo_p = st.session_state.get('conteo_actual', {'Healthy': 0, 'Anthracnose': 0, 'Scab': 0})
        
        st.metric(label="Total Procesadas", value=f"{total_p} und")
        st.metric(label="Fruta Comercial (Sana)", value=f"{conteo_p['Healthy']} und", delta="Apto para exportación")
        st.metric(label="Fruta Descarte (Infectada)", value=f"{conteo_p['Anthracnose'] + conteo_p['Scab']} und", delta="- Crítico", delta_color="inverse")
        
        st.divider()
        st.button("📄 Generar Reporte PDF")

    # =====================================================================
    # 4. SECCIÓN INFERIOR: Tablas de Clasificación y Exportación CSV
    # =====================================================================
    st.divider()
    st.subheader("📊 Desglose de Rendimiento y Auditoría de Datos")
    
    st.write("Filtro por Severidad de daño: **Mayor al 10%**")
    
    tab_sana, tab_antra, tab_rona = st.columns(3)
    
    with tab_sana:
        st.markdown("### 🟢 TABLA: SANA")
        df_sana = pd.DataFrame({"ID": ["P_01", "P_02"], "Calibre": [12, 14], "Efectividad %": [98, 95]})
        st.dataframe(df_sana, hide_index=True)
        st.download_button("Descargar CSV", data=df_sana.to_csv(index=False), file_name="sanas.csv", mime="text/csv")

    with tab_antra:
        st.markdown("### 🔴 TABLA: ANTRACNOSIS")
        df_antra = pd.DataFrame({"ID": ["P_03", "P_05"], "Daño Área": ["15%", "22%"], "Severidad": [1, 2]})
        st.dataframe(df_antra, hide_index=True)
        st.download_button("Descargar CSV", data=df_antra.to_csv(index=False), file_name="antracnosis.csv", mime="text/csv")

    with tab_rona:
        st.markdown("### 🟡 TABLA: ROÑA")
        df_rona = pd.DataFrame({"ID": ["P_07", "P_09"], "Daño Área": ["32%", "11%"], "Severidad": [2, 1]})
        st.dataframe(df_rona, hide_index=True)
        st.download_button("Descargar CSV", data=df_rona.to_csv(index=False), file_name="rona
