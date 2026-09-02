import cv2
import numpy as np
import streamlit as st
from PIL import Image


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
      return self.image

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

  def analyze_whitening_visual(self, warped_card):
    """Analizza i punti bianchi e disegna puntini rossi dove li trova."""
    output_img = warped_card.copy()
    h, w, _ = output_img.shape
    border_size = 20  

    # Aree dei 4 bordi
    regions = [
        ("top", 0, 0, w, border_size),
        ("bottom", 0, h - border_size, w, h),
        ("left", 0, 0, border_size, h),
        ("right", w - border_size, 0, w, h)
    ]

    total_white_pixels = 0

    for name, x1, y1, x2, y2 in regions:
      sub_region = output_img[y1:y2, x1:x2]
      gray_sub = cv2.cvtColor(sub_region, cv2.COLOR_BGR2GRAY)
      _, thresh = cv2.threshold(gray_sub, 200, 255, cv2.THRESH_BINARY)
      
      # Trova le coordinate dei pixel bianchi nella sotto-regione
      y_indices, x_indices = np.where(thresh > 200)
      total_white_pixels += len(x_indices)

      # Disegna puntini rossi sui pixel trovati
      for xi, yi in zip(x_indices, y_indices):
        # Mappa le coordinate locali a quelle globali dell'immagine
        global_x = x1 + xi
        global_y = y1 + yi
        cv2.circle(output_img, (global_x, global_y), 2, (0, 0, 255), -1)

    if total_white_pixels < 20:
      score = 10
    elif total_white_pixels < 60:
      score = 9
    elif total_white_pixels < 150:
      score = 8
    else:
      score = max(1, 8 - (total_white_pixels // 100))

    return output_img, score, total_white_pixels

  def analyze_centering_visual(self, warped_card):
    """Stima la centratura e disegna linee e metriche grafiche."""
    output_img = warped_card.copy()
    h, w, _ = output_img.shape

    # Ricerca del riquadro interno (artwork) tramite soglia di colore/grigio
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
      # Trova il rettangolo interno più plausibile (l'artwork o il box centrale)
      main_contour = max(contours, key=cv2.contourArea)
      x, y, cw, ch = cv2.boundingRect(main_contour)
      
      # Assicuriamoci che sia un box interno ragionevole
      if cw > w * 0.4 and ch > h * 0.4 and cw < w * 0.95 and ch < h * 0.95:
        # Calcolo margini in pixel
        left_margin = x
        right_margin = w - (x + cw)
        top_margin = y
        bottom_margin = h - (y + ch)

        # Disegna il rettangolo interno stimato
        cv2.rectangle(output_img, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
        
        # Linee guida visive
        cv2.line(output_img, (x, 0), (x, h), (255, 0, 0), 1)
        cv2.line(output_img, (x + cw, 0), (x + cw, h), (255, 0, 0), 1)
        cv2.line(output_img, (0, y), (w, y), (255, 0, 0), 1)
        cv2.line(output_img, (0, y + ch), (w, y + ch), (255, 0, 0), 1)

        # Calcolo percentuali di centratura approssimativa
        tot_lr = left_margin + right_margin
        left_pct = round((left_margin / tot_lr) * 100, 1) if tot_lr > 0 else 50
        right_pct = round(100 - left_pct, 1)

        tot_tb = top_margin + bottom_margin
        top_pct = round((top_margin / tot_tb) * 100, 1) if tot_tb > 0 else 50
        bottom_pct = round(100 - top_pct, 1)

        centering_text_lr = f"Sx/Dx: {left_pct}% / {right_pct}%"
        centering_text_tb = f"Up/Down: {top_pct}% / {bottom_pct}%"
        
        # Assegnazione voto centratura empirico basato su scostamento dal 50/50
        max_dev = max(abs(left_pct - 50), abs(top_pct - 50))
        if max_dev <= 2:
          centering_score = 10
        elif max_dev <= 5:
          centering_score = 9
        elif max_dev <= 10:
          centering_score = 8
        else:
          centering_score = 7

        return output_img, centering_score, centering_text_lr, centering_text_tb

    # Fallback se non rileva perfettamente il box interno
    return output_img, 9, "Sx/Dx: N/D", "Up/Down: N/D"


# Interfaccia Grafica Streamlit
st.title("🃏 TCG Card Grader (Analisi Avanzata)")
st.write("Carica la foto della carta per visualizzare i punti bianchi (in rosso) e le linee di centratura.")

uploaded_file = st.file_uploader("Scegli un'immagine...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  image_np = np.array(image)
  image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

  st.image(image, caption="Immagine originale", use_container_width=True)

  if st.button("Avvia Analisi Completa"):
    with st.spinner("Elaborazione geometrica e visiva in corso..."):
      grader = PokemonCardGrader(image_bgr)
      warped = grader.warp_card()

      # Esegui analisi punti bianchi con grafica
      img_white, edge_score, white_count = grader.analyze_whitening_visual(warped)
      
      # Esegui analisi centratura con grafica
      img_centering, centering_score, text_lr, text_tb = grader.analyze_centering_visual(warped)

      st.success("Analisi completata!")

      # Mostra metriche principali
      col1, col2 = st.columns(2)
      with col1:
        st.metric(label="Voto Bordi (Edges)", value=edge_score)
        st.write(f"Punti bianchi rilevati: {white_count}")
      with col2:
        st.metric(label="Voto Centratura", value=centering_score)
        st.write(text_lr)
        st.write(text_tb)

      # Mostra immagini elaborate side-by-side
      st.subheader("Visualizzazione Grafica dei Difetti")
      col_a, col_b = st.columns(2)
      
      with col_a:
        st.image(cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB), caption="Punti Bianchi (Punti Rossi)", use_container_width=True)
      with col_b:
        st.image(cv2.cvtColor(img_centering, cv2.COLOR_BGR2RGB), caption="Linee di Centratura", use_container_width=True)