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
import datetime # <-- NUEVA IMPORTACIÓN PARA FECHA Y HORA

# =====================================================================
# 1. CONFIGURACIÓN DEL DASHBOARD INDUSTRIAL (LAYOUT WIDE)
# =====================================================================
st.set_page_config(
    page_title="PaltosWeb - Visión IA", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- NUEVO: FONDO TEMÁTICO VERDE PALTA ---
st.markdown("""
<style>
    /* Aplicar fondo verde oscuro/noche al contenedor principal */
    .stApp {
        background-color: #0b1c11; /* Verde palta muy oscuro */
        background-image: radial-gradient(circle at 50% 0%, #153320 0%, #060e08 100%);
    }
    
    /* Para asegurar que las tablas y métricas sigan siendo legibles con el nuevo fondo */
    [data-testid="stMetricValue"] { color: #87e8a9; }
    div[data-testid="stTable"] { background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN TÉCNICA ---
device = torch.device("cpu")
class_names = ['Anthracnose', 'Healthy', 'Scab']
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
st.title("🥑 PaltoWeb - Visión IA")
pestana_monitoreo, pestana_historial, pestana_reportes = st.tabs([
    "📺 Monitoreo en Vivo", 
    "📜 Historial de Análisis", 
    "📊 Reportes y Rendimiento"
])

with pestana_monitoreo:
    col_izquierda, col_central, col_derecha = st.columns([1, 2, 1])
    
    # -----------------------------------------------------------------
    # PANEL IZQUIERDO: Hora, Fecha y Alerta de Enfermedades
    # -----------------------------------------------------------------
    with col_izquierda:
        # 1. Fecha y Hora en Vivo
        st.subheader("⏱️ Registro de Tiempo")
        ahora = datetime.datetime.now()
        st.write(f"**Fecha:** {ahora.strftime('%d / %m / %Y')}")
        st.write(f"**Hora:** {ahora.strftime('%H:%M:%S')}")
        st.divider()
        
        # 2. Tabla de Alerta de Enfermedades (Luces)
        st.subheader("🚨 Panel de Alertas")
        
        # Creamos un placeholder (espacio vacío) para poder actualizar esta tabla
        # automáticamente DESPUÉS de que la IA termine de analizar en el panel central.
        tabla_alertas_placeholder = st.empty()
        
        # Función para dibujar la tabla de luces basada en las detecciones actuales
        def actualizar_luces(conteo_enfermedades):
            # Lógica de las "Luces": Si hay al menos 1 detectado, se enciende la luz correspondiente.
            luz_anthracnose = "🔴 DETECTADO" if conteo_enfermedades.get('Anthracnose', 0) > 0 else "⚪ Inactivo"
            luz_scab = "🟡 DETECTADO" if conteo_enfermedades.get('Scab', 0) > 0 else "⚪ Inactivo"
            luz_healthy = "🟢 PRESENTE" if conteo_enfermedades.get('Healthy', 0) > 0 else "⚪ Inactivo"
            
            data_luces = {
                "Enfermedad": ["Anthracnose", "Scab", "Healthy"],
                "Luz de Alerta": [luz_anthracnose, luz_scab, luz_healthy]
            }
            # Pintamos la tabla en el placeholder
            tabla_alertas_placeholder.table(pd.DataFrame(data_luces))
            
        # Dibujamos la tabla por primera vez con datos vacíos o guardados de la sesión anterior
        conteo_inicial = st.session_state.get('conteo_actual', {'Anthracnose': 0, 'Scab': 0, 'Healthy': 0})
        actualizar_luces(conteo_inicial)

    # -----------------------------------------------------------------
    # PANEL CENTRAL: Visualizador de Captura (Visor General)
    # -----------------------------------------------------------------
    with col_central:
        st.subheader("🖼️ Captura de Inspección")
        archivo_subido = st.file_uploader("Cargar captura para análisis", type=["jpg", "jpeg", "png"])
        contenedor_imagen = st.empty()
        
        if archivo_subido is not None:
            imagen_pil = Image.open(archivo_subido).convert('RGB')
            orig_img = np.array(imagen_pil)
            orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR) 
            
            contenedor_imagen.image(imagen_pil, caption="Imagen original para inspección", use_container_width=True)
            
            if st.button("🚀 INICIAR PIPELINE DE ANÁLISIS UNIFICADO", type="primary", use_container_width=True):
                with st.spinner("Ejecutando Pipeline de Redes Neuronales..."):
                    models_ia = cargar_ecosistema_ia()
                    yolo_model = models_ia[0]
                    res_model, dense_model, eff_model, vgg_model, rf_model = models_ia[1], models_ia[2], models_ia[3], models_ia[4], models_ia[5]
                    
                    resultados_yolo = yolo_model(orig_img_bgr, conf=conf_threshold)[0]
                    boxes = resultados_yolo.boxes.xyxy.numpy().astype(int)
                    
                    if len(boxes) == 0:
                        st.warning("No se detectaron frutos con el umbral actual.")
                    else:
                        conteo_votado = {name: 0 for name in class_names}
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
                                pred_res = class_names[np.argmax(prob_res)]
                                
                                prob_dense = F.softmax(dense_model(tensor), dim=1).cpu().numpy()[0]
                                pred_dense = class_names[np.argmax(prob_dense)]
                                
                                prob_eff = F.softmax(eff_model(tensor), dim=1).cpu().numpy()[0]
                                pred_eff = class_names[np.argmax(prob_eff)]
                                
                                feat_vgg = vgg_model(tensor).cpu().numpy()
                                prob_rf = rf_model.predict_proba(feat_vgg)[0]
                                pred_rf = class_names[np.argmax(prob_rf)]
                                
                                prob_final = (prob_res + prob_dense + prob_eff + prob_rf) / 4.0
                                
                                predicted_idx = np.argmax(prob_final)
                                pred_consensuado = class_names[predicted_idx]
                                confidence_score = prob_final[predicted_idx] * 100
                                
                            conteo_votado[pred_consensuado] += 1
                            
                            detailed_results_list.append({
                                "Fruto": fruit_id, # Ajustado para la tabla de la derecha
                                "Diagnóstico": pred_consensuado, 
                                "Porcentaje": confidence_score,
                                "Diagnóstico Votado": f"{colors.get(pred_consensuado, (128, 128, 128))}: {pred_consensuado}",
                                "Confianza %": round(confidence_score, 1),
                                "ResNet18": pred_res,
                                "DenseNet121": pred_dense,
                                "EfficientNetB2": pred_eff,
                                "VGG16+RF": pred_rf
                            })
                            
                            color = colors.get(pred_consensuado, (255, 255, 255))
                            cv2.rectangle(orig_img_bgr, (x1, y1), (x2, y2), color, 3)
                            etiqueta = f"{pred_consensuado} {confidence_score:.1f}%"
                            cv2.putText(orig_img_bgr, etiqueta, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        
                        img_final_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
                        contenedor_imagen.image(img_final_rgb, caption="Diagnóstico en tiempo real completado", use_container_width=True)
                        
                        # --- GUARDAR EN MEMORIA ---
                        st.session_state['conteo_actual'] = conteo_votado
                        st.session_state['resultados_detalle'] = detailed_results_list
                        st.session_state['datos_detallados'] = detailed_results_list
                        
                        # --- ¡MAGIA!: ACTUALIZAR LAS LUCES DE LA IZQUIERDA AL INSTANTE ---
                        actualizar_luces(conteo_votado)
                        
                        st.success("Inspección finalizada con éxito.")

    # -----------------------------------------------------------------
    # PANEL DERECHO: Resumen Ejecutivo (Métricas Globales)
    # -----------------------------------------------------------------
    with col_derecha:
        st.subheader("📈 Resultados de Enfermedades")
        resultados_detalle = st.session_state.get('resultados_detalle', [])
        
        if not resultados_detalle:
            st.info("Esperando captura y análisis de la cámara...")
        else:
            df = pd.DataFrame(resultados_detalle)
            
            # Formateamos solo las columnas que queremos mostrar en esta tabla resumen
            df_resumen = df[["Fruto", "Diagnóstico", "Porcentaje"]]
            
            st.dataframe(
                df_resumen,
                column_config={
                    "Fruto": "ID Fruto",
                    "Diagnóstico": st.column_config.TextColumn("Condición"),
                    "Porcentaje": st.column_config.ProgressColumn(
                        "Confianza",
                        help="Nivel de certeza de la red neuronal",
                        format="%.2f %%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.divider()
            total_p = len(resultados_detalle)
            sanas = sum(1 for r in resultados_detalle if r['Diagnóstico'] == 'Healthy')
            enfermas = total_p - sanas
            
            st.write(f"**Total analizados:** {total_p} und")
            st.caption(f"🟢 Sanas: {sanas} | 🔴 Con afección: {enfermas}")
        
        st.divider()
        st.button("📄 Generar Reporte PDF")

    # =====================================================================
    # 5. SECCIÓN DE AUDITORÍA: TABLA DE PREDICCIONES INDIVIDUALES
    # =====================================================================
    detalles = st.session_state.get('datos_detallados', [])
    if detalles:
        st.divider()
        st.subheader("🕵️‍♂️ Sección de Auditoría: Desglose por Modelo")
        
        df_detalles = pd.DataFrame(detalles)
        # Filtramos las columnas originales para no repetir las de resumen
        cols_auditoria = ["Fruto", "Diagnóstico Votado", "Confianza %", "ResNet18", "DenseNet121", "EfficientNetB2", "VGG16+RF"]
        df_final = df_detalles[cols_auditoria].copy()
        
        def colorear(x):
            if '255, 0, 0' in x: return '🔴 Anthracnose'
            if '0, 255, 0' in x: return '🟢 Healthy'
            if '255, 255, 0' in x: return '🟡 Scab'
            return x
        
        df_final['Diagnóstico Votado'] = df_detalles['Diagnóstico Votado'].apply(colorear)
        
        st.dataframe(df_final, hide_index=True, use_container_width=True)
        st.download_button("Exportar Auditoría Full CSV", data=df_final.to_csv(index=False), file_name="auditoria.csv", mime="text/csv")
