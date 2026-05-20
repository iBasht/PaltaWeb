import streamlit as st
import datetime
import pandas as pd
from PIL import Image
import numpy as np
import cv2

# =====================================================================
# 1. CONFIGURACIÓN INICIAL Y CONTROL DE SESIÓN
# =====================================================================
st.set_page_config(
    page_title="PaltoSense", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Inicializar la variable de estado si no existe
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

# =====================================================================
# 2. ESTILOS CSS REPLICADOS DE TU IMAGEN (PALTO SENSE)
# =====================================================================
st.markdown("""
<style>
    /* Ocultar elementos nativos molestos de Streamlit */
    [data-testid="stHeader"] { visibility: hidden; height: 0px; }
    
    /* Fondo verde menta ultra claro de la imagen */
    .stApp { 
        background-color: #F1F5F2; 
    }

    /* Contenedor central flotante del Formulario (Tarjeta Blanca) */
    div.stForm {
        background-color: #FFFFFF !important;
        border: none !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05) !important;
        padding: 40px !important;
        max-width: 450px;
        margin: 0 auto;
    }

    /* Inputs de texto grisáceos redondeados */
    .stTextInput div div input {
        background-color: #E6ECE8 !important;
        border: none !important;
        border-radius: 10px !important;
        color: #2F4F4F !important;
        padding: 12px !important;
    }

    /* Etiquetas de los campos */
    .stTextInput label {
        color: #1F3A2B !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Botón Login Verde Sólido Redondeado */
    div.stForm button {
        background-color: #1E824C !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 0px !important;
        width: 100% !important;
        transition: all 0.3s ease;
    }
    div.stForm button:hover {
        background-color: #145A32 !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# INTERFAZ 1: PANTALLA DE REPLICADA LOGIN (SI NO ESTÁ AUTENTICADO)
# =====================================================================
if not st.session_state['autenticado']:
    
    # Espaciado superior para centrar verticalmente
    st.write("")
    st.write("")
    st.write("")
    
    # Contenedor del Logo de la planta y textos superiores (PaltoSense)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <div style="background-color: #1E824C; width: 60px; height: 60px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                <span style="color: white; font-size: 2rem; font-weight: bold;">🌱</span>
            </div>
            <h1 style="font-family: 'Georgia', serif; color: #153320; font-size: 3.2rem; margin: 0; font-weight: 700;">PaltoSense</h1>
            <p style="color: #617D6B; font-size: 1.1rem; margin-top: 5px; font-weight: 500; letter-spacing: 0.5px;">Precision Agriculture Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    # Formulario centralizado
    with st.form(key='formulario_login'):
        st.markdown("""
            <h3 style="color: #153320; margin-top: 0; font-size: 1.3rem; font-weight: 700;">Welcome Back</h3>
            <p style="color: #617D6B; font-size: 0.95rem; margin-bottom: 25px;">Enter your credentials to access the farm controls.</p>
        """, unsafe_allow_html=True)
        
        usuario = st.text_input("Username", placeholder="")
        contrasena = st.text_input("Password", type="password", placeholder="")
        
        # Al presionar el botón se ejecuta la validación
        boton_ingresar = st.form_submit_button(label="Login")
        
        if boton_ingresar:
            # Reemplaza aquí por el usuario y contraseña que desees
            if usuario == "admin" and contrasena == "palto2026":
                st.session_state['autenticado'] = True
                st.rerun() # Recarga la app para pasar al dashboard
            else:
                st.error("Credenciales incorrectas. Inténtalo de nuevo.")


# =====================================================================
# INTERFAZ 2: EL DASHBOARD COMPLETO (SI YA SE LOGEÓ CORRECTAMENTE)
# =====================================================================
else:
    # Cambiar dinámicamente estilos para la app una vez dentro
    st.markdown("""
        <style>
            /* Resetear fondos para la zona de la aplicación */
            .stApp { background-color: #F0FDF4; }
        </style>
    """, unsafe_allow_html=True)

    # Cabecera Superior del Dashboard
    st.markdown("""
        <div style="background-color: #1E824C; padding: 2rem 1rem 2.5rem 1rem; margin: -5rem -4rem 2rem -4rem; text-align: center; border-radius: 0 0 25px 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
            <h1 style="color: white; font-size: 2.8rem; margin: 0; font-family: 'Georgia', serif;">PaltoSense Dashboard</h1>
            <p style="color: #DCFCE7; font-size: 1rem; margin-top: 5px;">Hectárea 1 - Sistema IoT de Monitoreo Activo</p>
        </div>
    """, unsafe_allow_html=True)

    col_izq, col_centro, col_der = st.columns([1, 2, 1])

    with col_centro:
        # Botón para cerrar sesión de forma segura
        if st.button("🚪 Cerrar Sesión Industrial"):
            st.session_state['autenticado'] = False
            st.rerun()
            
        st.divider()
        
        # Muestra la hora y fecha
        ahora = datetime.datetime.now()
        st.markdown(f"<p style='text-align: center; color: #1E824C; font-weight: bold; font-size: 1.2rem;'>📅 {ahora.strftime('%d / %m / %Y')} &nbsp;&nbsp;|&nbsp;&nbsp; 🕒 {ahora.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
        
        # --- AQUÍ VA EL RESTO DEL PIPELINE DE DETECCIÓN E IA QUE YA TENÍAMOS ---
        st.info("¡Inicio de sesión exitoso! Aquí se procesará el análisis de redes neuronales y datos del ESP32.")
