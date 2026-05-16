"""
Detecção de ferrugem em HSV (OpenCV)
Autor: você :)
Uso:
  python rust_hsv.py --img caminho/para/imagem.jpg --out out/
Parâmetros úteis:
  --hmin 5 --hmax 30 --smin 80 --vmin 40   # afinam a segmentação
  --roi 100,200,1200,800                   # x,y,w,h (opcional)
"""

import cv2 as cv
import numpy as np
import os
import argparse
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="caminho da imagem")
    ap.add_argument("--out", default="out", help="pasta de saída")
    # Faixas padrão (OpenCV: H 0–179, S 0–255, V 0–255)
    ap.add_argument("--hmin", type=int, default=5, help="matiz mínima (ferrugem ~ laranja/marrom)")
    ap.add_argument("--hmax", type=int, default=30, help="matiz máxima")
    ap.add_argument("--smin", type=int, default=80, help="saturação mínima")
    ap.add_argument("--vmin", type=int, default=40, help="valor (brilho) mínimo")
    ap.add_argument("--roi", type=str, default=None, help="x,y,w,h (opcional)")
    ap.add_argument("--resize", type=int, default=0, help="largura para redimensionar (0=original)")
    return ap.parse_args()

def apply_clahe_on_v(hsv):
    # Equaliza ligeiramente o canal V para mitigar iluminação desigual
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_eq = clahe.apply(v)
    return cv.merge([h, s, v_eq])

def main():
    args = parse_args()
    img = cv.imread(args.img)
    assert img is not None, f"Não consegui ler {args.img}"

    if args.resize > 0:
        r = args.resize / img.shape[1]
        img = cv.resize(img, (args.resize, int(img.shape[0]*r)), interpolation=cv.INTER_AREA)

    # ROI opcional
    roi = None
    if args.roi:
        x,y,w,h = map(int, args.roi.split(","))
        roi = (x,y,w,h)
        x2, y2 = x+w, y+h
        img_roi = img[y:y2, x:x2].copy()
    else:
        img_roi = img.copy()

    # Pré-processamento leve
    img_blur = cv.GaussianBlur(img_roi, (5,5), 0)

    # HSV + leve correção no V
    hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    hsv = apply_clahe_on_v(hsv)

    # Segmentação de ferrugem (faixa ajustável)
    lower = np.array([args.hmin, args.smin, args.vmin], dtype=np.uint8)
    upper = np.array([args.hmax, 255, 255], dtype=np.uint8)
    mask = cv.inRange(hsv, lower, upper)

    # Limpeza morfológica
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)

    # Contornos e sobreposição
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    overlay = img_roi.copy()
    overlay[mask>0] = cv.addWeighted(img_roi[mask>0], 0.3, np.array([0,140,255], dtype=np.uint8), 0.7, 0)  # laranja BGR
    contoured = overlay.copy()
    cv.drawContours(contoured, contours, -1, (0,0,255), 2)

    # Métrica: % de área corroída (na ROI)
    rust_px = int(mask.sum() / 255)
    total_px = mask.size
    rust_pct = 100.0 * rust_px / total_px

    # Compor resultados em imagem única
    gray = cv.cvtColor(img_roi, cv.COLOR_BGR2GRAY)
    hsv_viz = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    grid_top = np.hstack([img_roi, hsv_viz, cv.cvtColor(gray, cv.COLOR_GRAY2BGR)])
    grid_bot = np.hstack([cv.cvtColor(mask, cv.COLOR_GRAY2BGR), overlay, contoured])
    grid = np.vstack([grid_top, grid_bot])

    # Texto com métrica
    cv.rectangle(grid, (10,10), (520,70), (255,255,255), -1)
    cv.putText(grid, f"Area corroida (ROI): {rust_pct:.2f}%", (20,55),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv.LINE_AA)

    # Saídas
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    base = Path(args.img).stem
    if roi:
        # recolocar overlay na imagem completa para registrar localização
        full_overlay = img.copy()
        x,y,w,h = roi
        full_overlay[y:y+h, x:x+w] = contoured
        cv.imwrite(str(outdir / f"{base}_overlay_full.png"), full_overlay)

    cv.imwrite(str(outdir / f"{base}_mask.png"), mask)
    cv.imwrite(str(outdir / f"{base}_overlay.png"), contoured)
    cv.imwrite(str(outdir / f"{base}_grid.png"), grid)

    print(f"[OK] Área corroída (ROI{' completa' if not roi else ''}): {rust_pct:.2f}%")
    print(f"[OK] Arquivos salvos em: {outdir.resolve()}")

if __name__ == "__main__":
    main()
