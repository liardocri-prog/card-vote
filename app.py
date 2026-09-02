import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="TCG Card Grader", page_icon="🃏", layout="wide"
)


class PokemonCardGrader:

  def __init__(self, image_np):
    self.image = image_np

  def order_points(self, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

  def warp_card(self):
    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_contour = None
    for c in contours:
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      if len(approx) == 4 and cv2.contourArea(c) > 5000:
        card_contour = approx
        break

    if card_contour is None:
      return cv2.resize(self.image, (714, 1000))

    pts = card_contour.reshape(4, 2)
    rect = self.order_points(pts)

    width, height = 714, 1000
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(self.image, M, (width, height))
    return warped

  def analyze_whitening(self, warped_card, threshold_val, border_thickness):
    output_img = warped_card.copy()
    h, w, _ = output_img.shape

    regions = {
        "top": (0, 0, w, border_thickness),
        "bottom": (0, h - border_thickness, w, h),
        "left": (0, 0, border_thickness, h),
        "right": (w - border_thickness, 0, w, h),
    }

    total_defects = 0

    for name, (x1, y1, x2, y2) in regions.items():
      sub = output_img[y1:y2, x1:x2]
      gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

      # Usa la soglia decisa dallo slider
      _, thresh = cv2.threshold(gray_sub, threshold_val, 255, cv2.THRESH_BINARY)

      contours, _ = cv2.findContours(
          thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
      )
      for c in contours:
        area = cv2.contourArea(c)
        if 0.5 <= area <= 150:  # Filtro dimensione difetto
          total_defects += 1
          M_c = cv2.moments(c)
          if M_c["m00"] > 0:
            cx = int(M_c["m10"] / M_c["m00"]) + x1
            cy = int(M_c["m01"] / M_c["m00"]) + y1
            cv2.circle(output_img, (cx, cy), 3, (0, 0, 255), -1)

    # Calcolo punteggio bordi basato sui difetti trovati
    if total_defects == 0:
      score = 10
    elif total_defects <= 3:
      score = 9
    elif total_defects <= 8:
      score = 8
    elif total_defects <= 15:
      score = 7
    else:
      score = max(1, 7 - (total_defects // 10))

    return output_img, score, total_defects

  def analyze_centering(self, warped_card, offset_x, offset_y):
    output_img = warped_card.copy()
    h, w, _ = output_img.shape

    # Usiamo i margini impostati dai cursori per definire il riquadro interno della carta
    x1 = offset_x
    x2 = w - offset_x
    y1 = offset_y
    y2 = h - offset_y

    left_margin = x1
    right_margin = w - x2
    top_margin = y1
    bottom_margin = h - y2

    tot_lr = left_margin + right_margin
    left_pct = round((left_margin / tot_lr) * 100, 1) if tot_lr > 0 else 50.0
    right_pct = round(100 - left_pct, 1)

    tot_tb = top_margin + bottom_margin
    top_pct = round((top_margin / tot_tb) * 100, 1) if tot_tb > 0 else 50.0
    bottom_pct = round(100 - top_pct, 1)

    # Disegno grafico pulito
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(output_img, (x1, 0), (x1, h), (0, 0, 255), 2)
    cv2.line(output_img, (x2, 0), (x2, h), (0, 0, 255), 2)
    cv2.line(output_img, (0, y1), (w, y1), (0, 0, 255), 2)
    cv2.line(output_img, (0, y2), (w, y2), (0, 0, 255), 2)

    max_dev = max(abs(left_pct - 50), abs(top_pct - 50))
    if max_dev <= 2:
      centering_score = 10
    elif max_dev <= 5:
      centering_score = 9
    elif max_dev <= 10:
      centering_score = 8
    else:
      centering_score = 7

    text_lr = f"Sx: {left_pct}% / Dx: {right_pct}%"
    text_tb = f"Up: {top_pct}% / Down: {bottom_pct}%"

    return output_img, centering_score, text_lr, text_tb


# Interfaccia Grafica Streamlit
st.title("🃏 TCG Card Grader Interattivo")
st.write(
    "Regola i parametri nella barra laterale se l'analisi automatica non è"
    " perfetta per la tua foto."
)

# Sidebar con i controlli manuali per calibrare l'app
st.sidebar.header("🎛️ Calibrazione Parametri")
threshold_slider = st.sidebar.slider(
    "Sensibilità Punti Bianchi (Soglia)", 150, 240, 190, 5
)
border_size_slider = st.sidebar.slider(
    "Spessore Bordo Analizzato (px)", 10, 50, 25, 1
)
margin_x_slider = st.sidebar.slider(
    "Margine Orizzontale Centratura (px)", 20, 100, 45, 1
)
margin_y_slider = st.sidebar.slider(
    "Margine Verticale Centratura (px)", 20, 100, 45, 1
)

uploaded_file = st.file_uploader(
    "Scegli un'immagine...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  image_np = np.array(image)
  image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

  st.image(image, caption="Immagine originale", use_container_width=True)

  if st.button("Esegui Analisi Tarata"):
    with st.spinner("Elaborazione in corso..."):
      grader = PokemonCardGrader(image_bgr)
      warped = grader.warp_card()

      img_white, edge_score, white_count = grader.analyze_whitening(
          warped, threshold_slider, border_size_slider
      )
      img_centering, centering_score, text_lr, text_tb = grader.analyze_centering(
          warped, margin_x_slider, margin_y_slider
      )

      st.success("Fatto!")

      col1, col2 = st.columns(2)
      with col1:
        st.metric(label="Voto Bordi (Edges)", value=edge_score)
        st.write(f"Punti bianchi trovati: {white_count}")
      with col2:
        st.metric(label="Voto Centratura", value=centering_score)
        st.write(f"**Orizzontale:** {text_lr}")
        st.write(f"**Verticale:** {text_tb}")

      st.subheader("Risultato Visivo")
      col_a, col_b = st.columns(2)
      with col_a:
        st.image(
            cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB),
            caption="Punti Bianchi (Regolabili da Sidebar)",
            use_container_width=True,
        )
      with col_b:
        st.image(
            cv2.cvtColor(img_centering, cv2.COLOR_BGR2RGB),
            caption="Centratura (Regolabile da Sidebar)",
            use_container_width=True,
        )
