import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import io



# Cargar configuración
load_dotenv()
AI_ENDPOINT = os.getenv("AI_SERVICE_ENDPOINT")
AI_KEY = os.getenv("AI_SERVICE_KEY")


# Crear cliente
credentials = CognitiveServicesCredentials(AI_KEY)
cv_client = ComputerVisionClient(AI_ENDPOINT, credentials)

st.set_page_config(page_title="Análisis de Imágenes con Azure", layout="wide")
st.title("🧠 Azure Computer Vision - Análisis de Imágenes")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # 1) Leer todos los bytes de la imagen
    image_bytes = uploaded_file.read()

    # 2) Crear la imagen de PIL *desde* esos bytes
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Imagen subida", use_container_width=True)

    with st.spinner("Analizando imagen..."):
        # 3) Volver a empaquetar los mismos bytes en un BytesIO para Azure
        analysis = cv_client.analyze_image_in_stream(
            image=io.BytesIO(image_bytes),
            visual_features=[
                VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.objects
            ]
        )

    # Descripción
    if analysis.description and analysis.description.captions:
        st.subheader("📝 Descripción")
        for caption in analysis.description.captions:
            st.write(f"> {caption.text} ({caption.confidence * 100:.1f}%)")

    # Etiquetas
    if analysis.tags:
        st.subheader("🏷️ Etiquetas")
        tags = [f"{tag.name} ({tag.confidence * 100:.1f}%)" for tag in analysis.tags]
        st.markdown(", ".join(tags))

    # Objetos
    if analysis.objects:
        st.subheader("📦 Objetos detectados")

        draw = ImageDraw.Draw(image)
        for obj in analysis.objects:
            r = obj.rectangle
            box = [(r.x, r.y), (r.x + r.w, r.y + r.h)]
            draw.rectangle(box, outline="cyan", width=3)
            draw.text((r.x, r.y), obj.object_property, fill="cyan")

            st.write(f"→ {obj.object_property} ({obj.confidence * 100:.1f}%)")

        st.image(image, caption="Imagen con objetos", use_column_width=True)
    else:
        st.warning("No se detectaron objetos.")