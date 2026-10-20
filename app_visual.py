import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

st.set_page_config(page_title="MetalTech | Inspeção Inteligente", page_icon="🔷", layout="wide")

st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain],[data-testid="stMainBlockContainer"]{background-color:#0B1220;color:#E7F0FF}
header[data-testid="stHeader"]{background-color:#0B1220}
[data-testid="stDecoration"]{display:none}
[data-testid="stToolbar"]{visibility:hidden;height:0}
.block-container{padding-top:2rem;padding-bottom:2rem}
h1,h2,h3,h4,p,label{color:#E7F0FF}
div[data-testid="stMetric"]{background-color:#111D30;border:1px solid #263B5A;padding:18px;border-radius:12px}
div[data-testid="stFileUploader"]{background-color:#111D30;border:1px dashed #3976B8;border-radius:12px;padding:16px}
.stButton>button{background-color:#245B96;color:#FFF;border:1px solid #3976B8;border-radius:8px;padding:.6rem 1.2rem;font-weight:600}
.stButton>button:hover{background-color:#3976B8;color:#FFF;border-color:#7DB7FF}
a{color:#7DB7FF}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


def carregar_imagem(arquivo):
    return np.array(Image.open(arquivo).convert("RGB"))



def processar_imagem(imagem_rgb, limiar_baixo=70, limiar_alto=180):
    imagem_bgr = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

    # Suavização e detecção de bordas
    borrada = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(borrada, limiar_baixo, limiar_alto)

    # Localiza contornos internos e externos
    contornos, _ = cv2.findContours(
        bordas.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Desenha os contornos em verde
    contornos_bgr = cv2.cvtColor(
        imagem_rgb.copy(), cv2.COLOR_RGB2BGR
    )
    cv2.drawContours(
        contornos_bgr, contornos, -1, (40, 210, 120), 1
    )
    contornos_rgb = cv2.cvtColor(
        contornos_bgr, cv2.COLOR_BGR2RGB
    )

    # Calcula a concentração local de bordas
    concentracao = cv2.blur(
        bordas.astype(np.float32) / 255.0,
        (31, 31)
    )

    # Destaca em vermelho as áreas com maior concentração
    valores = concentracao[concentracao > 0]
    mascara = np.zeros_like(bordas, dtype=np.uint8)

    if valores.size > 0:
        limite = np.percentile(valores, 90)
        mascara[concentracao >= limite] = 255

    # Suaviza a máscara para evitar marcações muito fragmentadas
    mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
    mascara = cv2.threshold(
        mascara, 80, 255, cv2.THRESH_BINARY
    )[1]

    # Sobrepõe vermelho na imagem original
    areas_rgb = imagem_rgb.copy()
    camada_vermelha = np.zeros_like(areas_rgb)
    camada_vermelha[:, :] = (255, 40, 40)

    selecionada = mascara > 0
    areas_rgb[selecionada] = (
        0.45 * areas_rgb[selecionada]
        + 0.55 * camada_vermelha[selecionada]
    ).astype(np.uint8)

    # Métricas
    total_bordas = int(np.count_nonzero(bordas))
    densidade_bordas = (
        total_bordas / bordas.size * 100
        if bordas.size > 0 else 0
    )

    maior_area = max(
        (cv2.contourArea(c) for c in contornos),
        default=0.0
    )

    circularidade = None
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(maior)
        perimetro = cv2.arcLength(maior, True)

        if area > 0 and perimetro > 0:
            circularidade = float(
                4 * np.pi * area / (perimetro ** 2)
            )

    return {
        "cinza": cinza,
        "bordas": bordas,
        "contornos": contornos_rgb,
        "areas_sinalizadas": areas_rgb,
        "total_bordas": total_bordas,
        "total_contornos": len(contornos),
        "densidade_bordas": float(densidade_bordas),
        "maior_area": float(maior_area),
        "circularidade": circularidade
    }



st.markdown("""
# 🔷 METALTECH
### INSPEÇÃO INTELIGENTE · VISÃO COMPUTACIONAL
---
""")
st.write(
    "Plataforma experimental para inspeção visual de peças industriais "
    "utilizando processamento de imagens com OpenCV."
)
st.info(
    "Esta versão destaca bordas e contornos. Essas medidas podem ajudar na "
    "inspeção, mas não confirmam automaticamente se uma peça tem defeito."
)

st.subheader("📂 Inspecionar uma peça")
arquivo = st.file_uploader(
    "Envie uma imagem da peça",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Use uma imagem nítida, com boa iluminação e a peça inteira visível.",
)

with st.expander("⚙️ Ajustes da análise"):
    col_a, col_b = st.columns(2)
    with col_a:
        limiar_baixo = st.slider("Sensibilidade — limite baixo", 0, 200, 70)
    with col_b:
        limiar_alto = st.slider("Sensibilidade — limite alto", 50, 300, 180)

if arquivo is not None:
    imagem = carregar_imagem(arquivo)
    st.image(imagem, caption="Imagem enviada", use_container_width=True)

    if st.button("🔍 Analisar imagem", use_container_width=True):
        inicio = time.perf_counter()
        dados = processar_imagem(imagem, limiar_baixo, limiar_alto)
        tempo = (time.perf_counter() - inicio) * 1000
        st.session_state["resultado_inspecao"] = {
            "imagem_original": imagem,
            "dados": dados,
            "tempo": tempo,
            "nome_arquivo": arquivo.name,
        }

if "resultado_inspecao" in st.session_state:
    resultado = st.session_state["resultado_inspecao"]
    dados = resultado["dados"]

    st.markdown("---")
    st.subheader("📊 Resultado da análise")
    st.caption(f"Arquivo analisado: {resultado['nome_arquivo']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bordas detectadas", f"{dados['total_bordas']:,}".replace(",", "."))
    c2.metric("Contornos externos", dados["total_contornos"])
    c3.metric("Densidade de bordas", f"{dados['densidade_bordas']:.2f}%")
    c4.metric("Tempo de processamento", f"{resultado['tempo']:.2f} ms")

    st.warning(
        "Análise visual concluída — revisão humana recomendada. Sem uma referência "
        "validada ou um modelo treinado com exemplos rotulados, não é confiável "
        "classificar esta peça como boa ou defeituosa."
    )

    
    col_original, col_processada = st.columns(2)

    with col_original:
        st.markdown("**Imagem original**")
        st.image(
            resultado["imagem_original"],
            use_container_width=True
        )

    with col_processada:
        st.markdown("**Contornos encontrados**")
        st.image(
            dados["contornos"],
            use_container_width=True
        )

    st.markdown("### 🔴 Regiões sinalizadas para inspeção")
    st.image(
        dados["areas_sinalizadas"],
        caption="As áreas vermelhas indicam maior concentração de bordas, não defeitos confirmados.",
        use_container_width=True
    )

    st.markdown("### 🔬 Detalhes do processamento")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Imagem em tons de cinza**")
        st.image(dados["cinza"], use_container_width=True)

    with col2:
        st.markdown("**Bordas detectadas — Canny**")
        st.image(dados["bordas"], use_container_width=True)

    m1, m2 = st.columns(2)

    m1.metric(
        "Área do maior contorno",
        f"{dados['maior_area']:.0f} px²"
    )

    m2.metric(
        "Circularidade do maior contorno",
        f"{dados['circularidade']:.3f}"
        if dados["circularidade"] is not None
        else "Indisponível"
    )

    st.caption(
        "As métricas dependem do enquadramento, iluminação, textura e formato da peça. "
        "Elas são indicadores descritivos, não uma porcentagem de defeito."
    )
else:
    st.info("Envie uma imagem e clique em **Analisar imagem** para começar.")

st.markdown("---")
st.caption("METALTECH · Projeto Integrador — Fábrica Inteligente")
