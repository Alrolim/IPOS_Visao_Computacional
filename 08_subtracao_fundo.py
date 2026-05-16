import cv2
import numpy as np
from ultralytics import YOLO

def remove_biggest_person_better(image_path="100.jpg",
                                 output_path="100_clean.jpg",
                                 model_path="yolov8n-seg.pt",
                                 conf=0.4,
                                 erode_px=3, close_px=7,
                                 ns_radius=5, telea_radius=9,
                                 feather_sigma=3):
    # Carrega imagem
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    # YOLOv8-seg
    model = YOLO(model_path)
    results = model(image_path, conf=conf, verbose=False)

    # Seleciona a MAIOR pessoa
    biggest_mask = None
    max_area = 0
    for r in results:
        if r.masks is None or r.boxes is None:
            continue
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        # usa polygono se existir
        has_xy = getattr(r.masks, "xy", None) is not None and len(r.masks.xy) == len(cls_ids)
        for i, cid in enumerate(cls_ids):
            if cid != 0:  # pessoa = 0 no COCO
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            if has_xy and r.masks.xy[i] is not None and len(r.masks.xy[i]) >= 3:
                pts = np.round(r.masks.xy[i]).astype(np.int32)
                cv2.fillPoly(m, [pts], 255)
            else:
                raw = r.masks.data[i].cpu().numpy()
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
                m = (raw > 0.5).astype(np.uint8) * 255

            area = int(m.sum() // 255)
            if area > max_area:
                max_area = area
                biggest_mask = m

    if biggest_mask is None:
        print("Nenhuma pessoa detectada. Salvando original.")
        cv2.imwrite(output_path, img)
        return

    # --- Refino de máscara ---
    # fecha buracos
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    mask = cv2.morphologyEx(biggest_mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    # erode para evitar capturar borda do corpo (reduz halo)
    if erode_px > 0:
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        mask = cv2.erode(mask, k_erode, iterations=1)

    # --- Inpainting em 2 estágios ---
    # 1) preserva estruturas (rachaduras/linhas)
    stage1 = cv2.inpaint(img, mask, ns_radius, cv2.INPAINT_NS)
    # 2) suaviza/“texturiza” a área preenchida
    stage2 = cv2.inpaint(stage1, mask, telea_radius, cv2.INPAINT_TELEA)

    # --- Feather na borda para mesclar melhor ---
    feather = cv2.GaussianBlur(mask, (0, 0), feather_sigma)
    alpha = (feather.astype(np.float32) / 255.0)[..., None]  # 3 canais
    out = (alpha * stage2 + (1 - alpha) * img).astype(np.uint8)

    cv2.imwrite(output_path, out)
    print(f"✅ Remoção com inpainting melhorado salva em: {output_path}")
