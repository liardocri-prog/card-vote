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

  def analyze_whitening_visual(self, warped_card):
    """Rileva i punti bianchi e segna cerchietti rossi precisi."""
    output_img = warped_card.copy()
    h, w, _ = output_img.shape
    border_size = 22  # Spessore del bordo analizzato

    regions = {
        "top": (0, 0, w, border_size),
        "bottom": (0, h - border_size, w, h),
        "left": (0, 0, border_size, h),
        "right": (w - border_size, 0, w, h),
    }

    total_defects = 0

    for name, (x1, y1, x2, y2) in regions.items():
      sub = output_img[y1:y2, x1:x2]
      gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

      # Soglia per isolare punti molto chiari (bianchi)
      _, thresh = cv2.threshold(gray_sub, 210, 255, cv2.THRESH_BINARY)

      # Rimuove il rumore microscopico
      kernel = np.ones((2, 2), np.uint8)
      thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

      # Trova i singoli difetti come contorni separati
      contours, _ = cv2.findContours(
          thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
      )
      for c in contours:
        area = cv2.contourArea(c)
        # Filtra per considerare solo veri punti bianchi (evita macchie giganti o singoli pixel spuri)
        if 1 <= area <= 100:
          total_defects += 1
          M_c = cv2.moments(c)
          if M_c["m00"] > 0:
            cx = int(M_c["m10"] / M_c["m00"]) + x1
            cy = int(M_c["m01"] / M_c["m00"]) + y1
            # Cerchietto rosso pulito e preciso
            cv2.circle(output_img, (cx, cy), 3, (0, 0, 255), 1)

    # Calcolo punteggio bordi
    if total_defects == 0:
      score = 10
    elif total_defects <= 2:
      score = 9
    elif total_defects <= 6:
      score = 8
    elif total_defects <= 12:
      score = 7
    else:
      score = max(1, 7 - (total_defects // 8))

    return output_img, score, total_defects

  def analyze_centering_visual(self, warped_card):
    """Calcola i margini interni e disegna linee guida di centratura professionali."""
    output_img = warped_card.copy()
    h, w, _ = output_img.shape

    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # Analisi dei profili di intensità per trovare la cornice interna della carta
    center_strip_h = gray[int(h * 0.3) : int(h * 0.7), :]
    col_sums = np.mean(center_strip_h, axis=0)

    center_strip_v = gray[:, int(w * 0.3) : int(w * 0.7)]
    row_sums = np.mean(center_strip_v, axis=1)

    def find_edge(profile, start_idx, end_idx, direction="left"):
      sub_p = profile[start_idx:end_idx]
      if len(sub_p) < 5:
        return 35  # fallback di default in pixel (~5%)
      diff = np.abs(np.diff(sub_p))
      if len(diff) == 0:
        return 35
      best_idx = np.argmax(diff)
      return start_idx + best_idx

    # Ricerca dei margini nei primi 100 pixel esterni
    left_margin = find_edge(col_sums, 10, 120, "left")
    right_margin = w - find_edge(col_sums, w - 120, w - 10, "right")
    top_margin = find_edge(row_sums, 10, 120, "top")
    bottom_margin = h - find_edge(row_sums, h - 120, h - 10, "bottom")

    # Sicurezza per evitare valori sballati
    if not left_margin or left_margin < 10:
      left_margin = 35
    if not right_margin or right_margin < 10:
      right_margin = 35
    if not top_margin or top_margin < 10:
      top_margin = 35
    if not bottom_margin or bottom_margin < 10:
      bottom_margin = 35

    # Calcolo percentuali
    tot_lr = left_margin + right_margin
    left_pct = round((left_margin / tot_lr) * 100, 1) if tot_lr > 0 else 50.0
    right_pct = round(100 - left_pct, 1)

    tot_tb = top_margin + bottom_margin
    top_pct = round((top_margin / tot_tb) * 100, 1) if tot_tb > 0 else 50.0
    bottom_pct = round(100 - top_pct, 1)

    # Coordinate del rettangolo interno della cornice
    x1, x2 = left_margin, w - right_margin
    y1, y2 = top_margin, h - bottom_margin

    # Disegno grafico pulito sulla foto
    cv2.rectangle(
        output_img, (x1, y1), (x2, y2), (0, 255, 0), 2
    )  # Rettangolo verde guida
    cv2.line(output_img, (x1, 0), (x1, h), (0, 0, 255), 2)  # Linea sinistra rossa
    cv2.line(output_img, (x2, 0), (x2, h), (0, 0, 255), 2)  # Linea destra rossa
    cv2.line(output_img, (0, y1), (w, y1), (0, 0, 255), 2)  # Linea superiore rossa
    cv2.line(
        output_img, (0, y2), (w, y2), (0, 0, 255), 2
    )  # Linea inferiore rossa

    # Valutazione voto centratura basato sulla deviazione dal 50/50
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
st.title("🃏 TCG Card Grader (Analisi Professionale)")
st.write(
    "Carica la foto della carta per analizzare con precisione i punti bianchi"
    " e la centratura geometrica."
)

uploaded_file = st.file_uploader(
    "Scegli un'immagine...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  image_np = np.array(image)
  image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

  st.image(image, caption="Immagine originale", use_container_width=True)

  if st.button("Avvia Analisi Completa"):
    with st.spinner("Elaborazione avanzata in corso..."):
      grader = PokemonCardGrader(image_bgr)
      warped = grader.warp_card()

      # Analisi difetti e centratura
      img_white, edge_score, white_count = grader.analyze_whitening_visual(
          warped
      )
      img_centering, centering_score, text_lr, text_tb = (
          grader.analyze_centering_visual(warped)
      )

      st.success("Analisi completata con successo!")

      # Metriche voti
      col1, col2 = st.columns(2)
      with col1:
        st.metric(label="Voto Bordi (Edges)", value=edge_score)
        st.write(f"Punti bianchi circoscritti: {white_count}")
      with col2:
        st.metric(label="Voto Centratura", value=centering_score)
        st.write(f"**Orizzontale:** {text_lr}")
        st.write(f"**Verticale:** {text_tb}")

      # Preview Grafica
      st.subheader("Dettaglio Grafico dell'Analisi")
      col_a, col_b = st.columns(2)

      with col_a:
        st.image(
            cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB),
            caption="Usura Bordi (Cerchietti Rossi)",
            use_container_width=True,
        )
      with col_b:
        st.image(
            cv2.cvtColor(img_centering, cv2.COLOR_BGR2RGB),
            caption="Schema Centratura (Linee e Cornice)",
            use_container_width=True,
        )