import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

PASTEL_COLORS = [
    "#c8c4ff", "#b8d4f0", "#f5c4e8", "#ffd4c8", "#ffe8b8",
    "#b8ebd4", "#d4c8ff", "#c4e0f8", "#f0d0f0", "#e8d8c8",
]

PASTEL_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: "Plus Jakarta Sans", system-ui, -apple-system, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fd;
    }

    .pastel-bg {
        position: fixed;
        inset: 0;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
    }

    .pastel-bg .blob {
        position: absolute;
        border-radius: 50%;
        filter: blur(88px);
        opacity: 0.42;
    }

    .pastel-bg .b1 {
        width: 520px; height: 520px;
        top: -140px; right: -100px;
        background: radial-gradient(circle, #8b85ff 0%, #c8c4ff 55%, transparent 72%);
    }

    .pastel-bg .b2 {
        width: 460px; height: 460px;
        bottom: -120px; left: -90px;
        background: radial-gradient(circle, #91b7de 0%, #d4e8f7 50%, transparent 70%);
    }

    .pastel-bg .b3 {
        width: 380px; height: 380px;
        top: 38%; left: 28%;
        background: radial-gradient(circle, #f09cd7 0%, #fce0f4 55%, transparent 72%);
        opacity: 0.28;
    }

    .pastel-bg .b4 {
        width: 340px; height: 340px;
        top: 12%; left: 8%;
        background: radial-gradient(circle, #ffb8a8 0%, #ffe8e0 55%, transparent 72%);
        opacity: 0.32;
    }

    .pastel-bg .b5 {
        width: 300px; height: 300px;
        bottom: 18%; right: 12%;
        background: radial-gradient(circle, #ffbc6d 0%, #fff0d4 55%, transparent 72%);
        opacity: 0.26;
    }

    .pastel-bg .b6 {
        width: 260px; height: 260px;
        top: 55%; right: 32%;
        background: radial-gradient(circle, #32bf8a 0%, #c8f0e0 55%, transparent 72%);
        opacity: 0.22;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 249, 253, 0.55);
        backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.62);
        backdrop-filter: blur(18px);
        border-right: 1px solid #e8edf7;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .main .block-container {
        position: relative;
        z-index: 1;
        padding-top: 2rem;
        max-width: 960px;
    }

    h1 {
        color: #0f1523 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        text-align: center;
        margin-bottom: 0.25rem !important;
    }

    .app-subtitle {
        text-align: center;
        color: #6b7a99;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    [data-testid="stTabs"] button {
        border-radius: 16px 16px 0 0 !important;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.72);
        border: 1px dashed #c8c4ff;
        border-radius: 24px;
        backdrop-filter: blur(8px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #635bff, #8b85ff) !important;
        border: none !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 24px rgba(99, 91, 255, 0.28);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(99, 91, 255, 0.36);
    }

    [data-testid="stAlert"] {
        border-radius: 16px;
        backdrop-filter: blur(8px);
    }

    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]) {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid #e8edf7;
        border-radius: 24px;
        padding: 0.5rem 1rem 1rem;
        backdrop-filter: blur(10px);
    }
</style>

<div class="pastel-bg" aria-hidden="true">
    <span class="blob b1"></span>
    <span class="blob b2"></span>
    <span class="blob b3"></span>
    <span class="blob b4"></span>
    <span class="blob b5"></span>
    <span class="blob b6"></span>
</div>
"""

st.markdown(PASTEL_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Настройки")
    API_URL = st.text_input(
        "API URL",
        value="https://capi-api-o5nq.onrender.com/predict",
        help="URL вашего FastAPI бэкенда",
    )

st.title("Классификатор изображений CIFAR-10")
st.markdown(
    '<p class="app-subtitle">Загрузите фото или нарисуйте объект — модель определит класс</p>',
    unsafe_allow_html=True,
)


def send_to_api(image_bytes: bytes) -> dict | None:
    with st.spinner("Отправка на сервер (первый запрос может занять до 2 минут)..."):
        try:
            resp = requests.post(
                API_URL,
                files={"file": ("image.png", image_bytes, "image/png")},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API. Проверьте URL и доступность сервера.")
        except Exception as e:
            st.error(f"Ошибка: {e}")
    return None


def show_result(result: dict):
    st.success(
        f"**Предсказание:** {result['predicted_class']}  "
        f"(**{result['confidence']}%** уверенность)"
    )
    probs = result["probabilities"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=PASTEL_COLORS[: len(probs)],
                text=[f"{v}%" for v in probs.values()],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Распределение вероятностей по классам",
        yaxis_title="Вероятность (%)",
        yaxis_range=[0, 105],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f1523"),
        title_font=dict(size=16, color="#0f1523"),
    )
    st.plotly_chart(fig, use_container_width=True)


tab_upload, tab_canvas = st.tabs(["📷 Загрузить изображение", "✏️ Нарисовать на холсте"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Загруженное изображение", width=300)
        if st.button("Классифицировать", type="primary", key="btn_upload"):
            result = send_to_api(uploaded_file.getvalue())
            if result:
                show_result(result)

with tab_canvas:
    st.markdown("Нарисуйте изображение на холсте и нажмите **Классифицировать**.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=st.slider("Толщина кисти", 1, 30, 10),
        stroke_color=st.color_picker("Цвет кисти", "#635bff"),
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Классифицировать рисунок", type="primary", key="btn_canvas"):
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype(np.uint8)
            pil_img = Image.fromarray(img_array).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            result = send_to_api(buf.getvalue())
            if result:
                show_result(result)
        else:
            st.warning("Сначала что-нибудь нарисуйте на холсте.")
