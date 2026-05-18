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

# =====================================================================
# 1. CONFIGURACIÓN DEL DASHBOARD INDUSTRIAL (LAYOUT WIDE)
# =====================================================================
st.set_page_config(
    page_title="PaltosWeb - Visión IA", # Re-branding
    layout="wide", # ¡Modo Pantalla Completa!
    initial_sidebar_state="expanded"
)

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
    # Cargar el YOLO pequeño directamente desde GitHub (pesa poco)
    yolo_detector = YOLO('yolo26n.pt') 
    
    # --- RUTAS LOCALES DE PESOS ---
    path_res = 'modelo_resnet18_paltos.pth'
    path_dense = 'modelo_densenet_paltos.pth'
    path_eff = 'modelo_efficientnet_paltos.pth'
    path_rf = 'modelo_RF_VGG16.pkl'
    
    # ⚠️ REEMPLAZA AQUÍ TUS ENLACES DE GITHUB RELEASES ⚠️
    url_res = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_resnet18_paltos.pth"
    url_dense = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_DenseNet_paltos.pth"
    url_eff = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_EfficientNet_paltos.pth"
    url_rf = "https://github.com/iBasht/PaltaWeb/releases/download/v1.0/modelo_RF_VGG16.pkl"
    
    # --- FUNCIÓN INTELIGENTE DE DESCARGA ---
    def descargar_cerebro(path, url, nombre_modelo):
        if not os.path.exists(path):
            with st.spinner(f"Sincronizando {nombre_modelo} con el servidor cloud..."):
                urllib.request.urlretrieve(url, path)

    # Descargar todos los cerebros pesados
    descargar_cerebro(path_res, url_res, "ResNet18")
    descargar_cerebro(path_dense, url_dense, "DenseNet121")
    descargar_cerebro(path_eff, url_eff, "EfficientNetB2")
    descargar_cerebro(path_rf, url_rf, "VGG16+RF")
    
    # --- CONFIGURACIÓN DE ARQUITECTURAS E IMÁGENES ---
    
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
    
    # VGG16 + Random Forest (Híbrido)
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
# 3. BARRA LATERAL (SIDEBAR) - CONTROL DE USUARIO Y CONFIGURACIÓN
# =====================================================================
with st.sidebar:
    st.title("⚙️ Configuración")
    st.write("**Usuario Activo:** Sebastian")
    st.write("**Empresa:** PaltosWeb - Visión IA") # Re-branding
    st.divider()
    
    # Control deslizante para el umbral de YOLO (Pedido en tu boceto)
    conf_threshold = st.slider("Umbral de Confianza YOLO", min_value=0.10, max_value=1.00, value=0.50, step=0.05)
    st.caption("Ajusta la sensibilidad para la detección de los frutos en campo.")
    st.divider()
    if st.button("Cerrar Sesión", type="secondary"):
        st.info("Sesión finalizada.")

# =====================================================================
# 4. CUERPO PRINCIPAL DEL DASHBOARD: TABS Y PANELES
# =====================================================================
st.title("🥑 PaltoWeb - Visión IA") # Nombre de producto/sistema
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
    # PANEL IZQUIERDO: KPI de Lote y Monitoreo Individual
    # -----------------------------------------------------------------
    with col_izquierda:
        st.subheader("📋 Estado de Lote")
        st.metric(label="Lote Actual", value="LOTE-2026A", delta="Fundo Trujillo")
        
        st.write("**Monitoreo Individual del Lote:**")
        # Tabla dinámica simulada para la barra lateral izquierda
        data_monitoreo = {
            "ID Fruto": ["Palta_042", "Palta_043", "Palta_044"],
            "Diagnóstico Votado": ["🔴 Anthracnose", "🟢 Healthy", "🟡 Scab"]
        }
        st.table(pd.DataFrame(data_monitoreo))

    # -----------------------------------------------------------------
    # PANEL CENTRAL: Visualizador de Captura (Visor General)
    # -----------------------------------------------------------------
    with col_central:
        st.subheader("🖼️ Captura de Inspección") # Nombre general solicitado
        archivo_subido = st.file_uploader("Cargar captura para análisis", type=["jpg", "jpeg", "png"])
        
        # Marcadores de posición para las imágenes
        contenedor_imagen = st.empty()
        
        if archivo_subido is not None:
            imagen_pil = Image.open(archivo_subido).convert('RGB')
            orig_img = np.array(imagen_pil)
            orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR) 
            
            contenedor_imagen.image(imagen_pil, caption="Imagen original para inspección", use_container_width=True)
            
            if st.button("🚀 INICIAR PIPELINE DE ANÁLISIS UNIFICADO", type="primary", use_container_width=True):
                with st.spinner("Ejecutando Pipeline de Redes Neuronales..."):
                    # Cargar ecosistema completo
                    models = cargar_ecosistema_ia()
                    yolo_model = models[0]
                    res_model, dense_model, eff_model, vgg_model, rf_model = models[1], models[2], models[3], models[4], models[5]
                    
                    # 1. Cazador YOLO (con umbral del slider)
                    resultados_yolo = yolo_model(orig_img_bgr, conf=conf_threshold)[0]
                    boxes = resultados_yolo.boxes.xyxy.numpy().astype(int)
                    
                    if len(boxes) == 0:
                        st.warning("No se detectaron frutos con el umbral actual.")
                    else:
                        conteo_votado = {name: 0 for name in class_names}
                        detailed_results_list = [] # Para la tabla de auditoría final
                        
                        # 2. Iteración y clasificación por el ecosistema
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(orig_img_bgr.shape[1], box[2]), min(orig_img_bgr.shape[0], box[3])
                            recorte = orig_img_bgr[y1:y2, x1:x2]
                            recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
                            pil_recorte = Image.fromarray(recorte_rgb)
                            
                            tensor = ecosistema(pil_recorte).unsqueeze(0)
                            fruit_id = f"P_{i+1:03d}"
                            
                            with torch.no_grad():
                                # Predicciones Individuales
                                # Probabilidades ResNet
                                prob_res = F.softmax(res_model(tensor), dim=1).cpu().numpy()[0]
                                pred_res = class_names[np.argmax(prob_res)]
                                
                                # Probabilidades DenseNet
                                prob_dense = F.softmax(dense_model(tensor), dim=1).cpu().numpy()[0]
                                pred_dense = class_names[np.argmax(prob_dense)]
                                
                                # Probabilidades EfficientNet
                                prob_eff = F.softmax(eff_model(tensor), dim=1).cpu().numpy()[0]
                                pred_eff = class_names[np.argmax(prob_eff)]
                                
                                # Probabilidades VGG16+RF (Híbrido)
                                feat_vgg = vgg_model(tensor).cpu().numpy()
                                prob_rf = rf_model.predict_proba(feat_vgg)[0]
                                pred_rf = class_names[np.argmax(prob_rf)]
                                
                                # EL CONSENSO (Promedio de las 4 opiniones)
                                prob_final = (prob_res + prob_dense + prob_eff + prob_rf) / 4.0
                                
                                # Seleccionar al ganador
                                predicted_idx = np.argmax(prob_final)
                                pred_consensuado = class_names[predicted_idx]
                                confidence_score = prob_final[predicted_idx] * 100
                                
                            conteo_votado[pred_consensuado] += 1
                            
                            # Guardar datos para la tabla de auditoría inferior
                            detailed_results_list.append({
                                "Fruit ID": fruit_id,
                                "Diagnóstico Votado": f"{colors.get(pred_consensuado, (128, 128, 128))}: {pred_consensuado}", # Usamos un truco de texto para simular color en pandas
                                "Confianza %": round(confidence_score, 1),
                                "ResNet18": pred_res,
                                "DenseNet121": pred_dense,
                                "EfficientNetB2": pred_eff,
                                "VGG16+RF": pred_rf
                            })
                            
                            # Dibujar cuadro con el label consensuado
                            color = colors.get(pred_consensuado, (255, 255, 255))
                            cv2.rectangle(orig_img_bgr, (x1, y1), (x2, y2), color, 3)
                            etiqueta = f"{pred_consensuado} {confidence_score:.1f}%"
                            cv2.putText(orig_img_bgr, etiqueta, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        
                        # Reemplazar imagen original con la procesada por la IA
                        img_final_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
                        contenedor_imagen.image(img_final_rgb, caption="Diagnóstico en tiempo real completado", use_container_width=True)
                        
                        # Guardar en el estado de Streamlit para usarlo en los otros paneles
                        st.session_state['conteo_actual'] = conteo_votado
                        st.session_state['total_paltas'] = len(boxes)
                        st.session_state['datos_detallados'] = detailed_results_list # Nueva lista para la auditoría
                        st.success("Inspección finalizada con éxito.")

    # -----------------------------------------------------------------
    # PANEL DERECHO: Resumen Ejecutivo (Métricas Globales)
    # -----------------------------------------------------------------
    with col_derecha:
        st.subheader("📈 Resultados de Enfermedades")
        
        # Recuperamos la lista de resultados detallados desde el estado de la sesión.
        # (Asegúrate de guardar los resultados de tu modelo en esta variable)
        resultados_detalle = st.session_state.get('resultados_detalle', [])
        
        if not resultados_detalle:
            st.info("Esperando captura y análisis de la cámara...")
        else:
            import pandas as pd
            df = pd.DataFrame(resultados_detalle)
            
            # Configuramos la tabla para mostrar la barra de porcentaje visualmente
            st.dataframe(
                df,
                column_config={
                    "Fruto": "ID Fruto",
                    "Diagnóstico": st.column_config.TextColumn("Condición"),
                    "Porcentaje": st.column_config.ProgressColumn(
                        "Confianza del Modelo",
                        help="Nivel de certeza de la red neuronal",
                        format="%.2f %%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Resumen del lote basado en tu consola de Colab
            st.divider()
            total_p = len(resultados_detalle)
            sanas = sum(1 for r in resultados_detalle if r['Diagnóstico'] == 'Healthy')
            enfermas = total_p - sanas
            
            st.write(f"**Total analizados:** {total_p} und")
            st.caption(f"🟢 Sanas: {sanas} | 🔴 Con afección: {enfermas}")
        
        st.divider()
        st.button("📄 Generar Reporte PDF")

    # =====================================================================
    # 5. NUEVA SECCIÓN DE AUDITORÍA: TABLA DE PREDICCIONES INDIVIDUALES
    # =====================================================================
    # Se despliega solo si hay datos analizados
    detalles = st.session_state.get('datos_detallados', [])
    if detalles:
        st.divider()
        st.subheader("🕵️‍♂️ Sección de Auditoría: Desglose de Predicciones por Modelo")
        
        # Convertimos la lista de diccionarios en un DataFrame de Pandas
        df_detalles = pd.DataFrame(detalles)
        
        # Truco para que las etiquetas de color se vean bien en pandas (simulando tu boceto)
        # Reemplazamos el texto por una versión más amigable
        df_final = df_detalles.copy()
        
        # Usamos emojis como marcadores de color para que Streamlit los renderice
        def colorear(x):
            if '255, 0, 0' in x: return '🔴 Antracnosis'
            if '0, 255, 0' in x: return '🟢 Healthy'
            if '255, 255, 0' in x: return '🟡 Scab'
            return x
        
        df_final['Diagnóstico Votado'] = df_detalles['Diagnóstico Votado'].apply(colorear)
        
        # Mostrar la tabla de auditoría completa de forma limpia
        st.dataframe(df_final, hide_index=True, use_container_width=True)
        
        # Botón de descarga de auditoría completa
        st.download_button("Exportar Auditoría Full CSV", data=df_final.to_csv(index=False), file_name="auditoria.csv", mime="text/csv")
