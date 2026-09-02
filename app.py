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

  def analyze_whitening(self, warped_card):
    h, w, _ = warped_card.shape
    border_size = 15
    top_b = warped_card[:border_size, :]
    bottom_b = warped_card[h - border_size :, :]
    left_b = warped_card[:, :border_size]
    right_b = warped_card[:, w - border_size :]

    borders = [top_b, bottom_b, left_b, right_b]
    total_white_pixels = 0

    for b in borders:
      gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
      _, thresh = cv2.threshold(gray_b, 200, 255, cv2.THRESH_BINARY)
      total_white_pixels += np.count_nonzero(thresh > 200)

    if total_white_pixels < 20:
      score = 10
    elif total_white_pixels < 60:
      score = 9
    elif total_white_pixels < 150:
      score = 8
    else:
      score = max(1, 8 - (total_white_pixels // 100))

    return score, total_white_pixels

  def grade_card(self):
    warped = self.warp_card()
    edge_score, white_count = self.analyze_whitening(warped)
    return {
        "Estimated_PSA_Grade": edge_score,
        "Edges_Score": edge_score,
        "White_Pixels_Detected": white_count,
    }


# Interfaccia Grafica Streamlit
st.title("🃏 TCG Card Grader (Pokémon & One Piece)")
st.write(
    "Carica la foto di una carta per stimare lo stato di conservazione"
    " analizzando i bordi."
)

uploaded_file = st.file_uploader(
    "Scegli un'immagine...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  image_np = np.array(image)
  image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

  st.image(image, caption="Immagine caricata", use_container_width=True)

  if st.button("Avvia Analisi"):
    with st.spinner("Analisi della carta in corso..."):
      grader = PokemonCardGrader(image_bgr)
      result = grader.grade_card()

      st.success("Analisi completata!")
      st.metric(label="Voto PSA Stimato", value=result["Estimated_PSA_Grade"])
      st.write(
          f"**Pixel bianchi rilevati sui bordi:**"
          f" {result['White_Pixels_Detected']}"
      )