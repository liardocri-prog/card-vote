import cv2
import numpy as np


class PokemonCardGrader:

  def __init__(self, image_path):
    self.image = cv2.imread(image_path)
    if self.image is None:
      raise ValueError(f"Impossibile caricare l'immagine dal percorso: {image_path}")

  def order_points(self, pts):
    """Ordina i 4 punti del contorno della carta: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left ha la somma minima
    rect[2] = pts[np.argmax(s)]  # Bottom-right ha la somma massima

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right ha la differenza minima
    rect[3] = pts[np.argmax(diff)]  # Bottom-left ha la differenza massima
    return rect

  def warp_card(self):
    """Indole la carta nello sfondo, la raddrizza e la ridimensiona a una misura standard."""
    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Trova i contorni
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_contour = None
    for c in contours:
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      # Se il contorno ha 4 lati ed è abbastanza grande, è la carta
      if len(approx) == 4 and cv2.contourArea(c) > 5000:
        card_contour = approx
        break

    if card_contour is None:
      print(
          "Avviso: Bordo carta non rilevato automaticamente. Uso l'intera"
          " immagine."
      )
      return self.image

    # Trasformazione prospettica (Raddrizzamento)
    pts = card_contour.reshape(4, 2)
    rect = self.order_points(pts)

    # Standard formato carta collezionabile (es. proporzione 714x1000 pixel)
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
    """Analizza i bordi della carta per individuare punti bianchi (usura)."""
    h, w, _ = warped_card.shape
    border_size = 15  # Spessore del bordo da analizzare in pixel

    # Estrai i quattro bordi
    top_b = warped_card[:border_size, :]
    bottom_b = warped_card[h - border_size :, :]
    left_b = warped_card[:, :border_size]
    right_b = warped_card[:, w - border_size :]

    borders = [top_b, bottom_b, left_b, right_b]
    total_white_pixels = 0

    for b in borders:
      gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
      # Consideriamo "punto bianco" un pixel con luminosità molto alta (da tarare in base al colore del retro)
      # Supponendo il retro blu tipico dei Pokémon, i punti bianchi spiccano per alta intensità
      _, thresh = cv2.threshold(gray_b, 200, 255, cv2.THRESH_BINARY)
      total_white_pixels += cv.countNonZero(thresh) if "cv" in globals() else np.count_nonzero(thresh > 200)

    # Assegnazione di un punteggio empirico basato sui difetti trovati
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

    # Simulazione logica di valutazione complessiva (stile PSA)
    # Nel grading reale si valutano Centering, Corners, Edges, Surface.
    final_estimate = edge_score

    return {
        "Estimated_PSA_Grade": final_estimate,
        "Edges_Score": edge_score,
        "White_Pixels_Detected": white_count,
    }


# Esempio di utilizzo:
# grader = PokemonCardGrader("percorso_foto_carta.jpg")
# risultato = grader.grade_card()
# print(risultato)