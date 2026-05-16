"""
Remoção de artefatos preservando trincas (Pré-processamento)
Autor: Adriana Rolim
Requisitos: opencv-python, numpy

Ideia:
1) Detectar trincas como linhas escuras finas via MORPH_BLACKHAT.
2) Criar uma máscara da trinca e preservar essa região sem suavização.
3) Aplicar suavização forte (NLMeans + Bilateral) apenas no fundo.
4) Recombinar e salvar 'antes/depois' lado a lado.

Uso:
  python remove_artefatos_preservando_trinca.py --img parede_trinca.jpg --out out/
Opções úteis:
  --maxw 1200                # redimensiona para largura máx (0=sem resize)
  --bh_ksize 21              # kernel do blackhat (ímpar > 1)
  --dilate 1                 # dilata a máscara da trinca
  --nlm_h 10 --bil_d 11 --bil_sc 75 --bil_ss 75  # força da suavização
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="caminho da imagem")
    ap.add_argument("--out", default="out", help="pasta de saída")
    ap.add_argument("--maxw", type=int, default=1200, help="largura máx. p/ processamento (0=original)")
    ap.add_argument("--bh_ksize", type=int, default=21, help="kernel (ímpar) do blackhat")
    ap.add_argument("--dilate", type=int, default=1, help="iterações de dilatação da máscara da trinca")
    # Parâmetros da suavização
    ap.add_argument("--nlm_h", type=int, default=10, help="h do fastNlMeansDenoisingColored")
    ap.add_argument("--nlm_template", type=int, default=7, help="templateWindowSize do NLM")
    ap.add_argument("--nlm_search", type=int, default=21, help="searchWindowSize do NLM")
    ap.add_argument("--bil_d", type=int, default=11, help="d do bilateral")
    ap.add_argument("--bil_sc", type=int, default=75, help="sigmaColor do bilateral")
    ap.add_argument("--bil_ss", type=int, default=75, help="sigmaSpace do bilateral")
    return ap.parse_args()

def detect_crack_mask(img_bgr, ksize=21, dilate_iters=1):
    """Cria máscara binária da trinca usando Black-Hat (linhas escuras finas)."""
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)

    # Kernel elíptico grande para captar fundo e realçar linhas escuras
    ksize = max(3, ksize | 1)  # garante ímpar >=3
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, se)

    # Limiar (Otsu) para obter máscara
    _, mask = cv.threshold(blackhat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Dilata levemente para cobrir a trinca por completo
    if dilate_iters > 0:
        mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), iterations=dilate_iters)
    return mask

def smooth_background_only(img_bgr, crack_mask, nlm_h=10, nlm_template=7, nlm_search=21,
                           bil_d=11, bil_sc=75, bil_ss=75):
    """Suaviza fundo (fora da máscara) e preserva trinca."""
    # Denoising forte + bilateral (bordas preservadas)
    den = cv.fastNlMeansDenoisingColored(img_bgr, None, nlm_h, nlm_h, nlm_template, nlm_search)
    sm = cv.bilateralFilter(den, d=bil_d, sigmaColor=bil_sc, sigmaSpace=bil_ss)

    bg_mask = cv.bitwise_not(crack_mask)
    bg_smoothed = cv.bitwise_and(sm, sm, mask=bg_mask)
    fg_crack   = cv.bitwise_and(img_bgr, img_bgr, mask=crack_mask)
    result = cv.add(bg_smoothed, fg_crack)
    return result

def main():
    args = parse_args()
    inp = Path(args.img)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    img = cv.imread(str(inp))
    assert img is not None, f"Não consegui ler {inp}"

    # Resize opcional (mantém proporção)
    if args.maxw and args.maxw > 0 and img.shape[1] > args.maxw:
        r = args.maxw / img.shape[1]
        img = cv.resize(img, (args.maxw, int(img.shape[0]*r)), interpolation=cv.INTER_AREA)

    crack_mask = detect_crack_mask(img, ksize=args.bh_ksize, dilate_iters=args.dilate)
    result = smooth_background_only(
        img, crack_mask,
        nlm_h=args.nlm_h, nlm_template=args.nlm_template, nlm_search=args.nlm_search,
        bil_d=args.bil_d, bil_sc=args.bil_sc, bil_ss=args.bil_ss
    )

    # Saídas
    base = inp.stem
    cv.imwrite(str(outdir / f"{base}_crack_mask.png"), crack_mask)
    cv.imwrite(str(outdir / f"{base}_clean.png"), result)
    side = cv.hconcat([img, result])
    cv.imwrite(str(outdir / f"{base}_comparativo.png"), side)

    print("[OK] Arquivos salvos em:", outdir.resolve())
    print("  - Máscara da trinca:", f"{base}_crack_mask.png")
    print("  - Imagem limpa:", f"{base}_clean.png")
    print("  - Comparativo:", f"{base}_comparativo.png")

if __name__ == "__main__":
    main()
