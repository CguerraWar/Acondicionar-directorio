import os
import cv2
import numpy as np

ROOT_DIR = r"C:\Users\table\Documents\Carlos\Personales\B\Zapatos"
HD_FOLDER_NAME = "HD"

ALPHA_THRESHOLD = 1

# Slot meaning:
# 1 = pair crossed (both shoes)
# 2 = right view
# 3 = left view
# 4 = diagonal/angle
# 5 = sole


def read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        bgr = img
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([bgr, a], axis=2)
    return img


def alpha_mask(rgba):
    a = rgba[:, :, 3]
    m = (a > ALPHA_THRESHOLD).astype(np.uint8) * 255
    return m


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 + 1, y1 + 1)


def bbox_fill_ratio(mask):
    bb = bbox_from_mask(mask)
    if bb is None:
        return 0.0
    x0, y0, x1, y1 = bb
    box_area = float((x1 - x0) * (y1 - y0))
    if box_area <= 0.0:
        return 0.0
    obj_area = float((mask > 0).sum())
    return obj_area / box_area


def connected_components_areas(mask):
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if num <= 1:
        return []
    areas = []
    for i in range(1, num):
        areas.append(int((labels == i).sum()))
    areas.sort(reverse=True)
    return areas


def pca_angle_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    c = pts - mean
    cov = np.cov(c.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    return ang


def rotate_mask(mask, angle_deg):
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot


def height_width_ratio_aligned(mask):
    ang = pca_angle_from_mask(mask)
    rot = rotate_mask(mask, -ang)
    bb = bbox_from_mask(rot)
    if bb is None:
        return 0.0
    x0, y0, x1, y1 = bb
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    return float(h) / float(w)


def facing_direction_aligned(mask):
    ang = pca_angle_from_mask(mask)
    rot = rotate_mask(mask, -ang)
    bb = bbox_from_mask(rot)
    if bb is None:
        return 0
    x0, y0, x1, y1 = bb
    obj = rot[y0:y1, x0:x1]
    h, w = obj.shape[:2]
    if w < 50 or h < 50:
        return 0

    obj_bin = (obj > 0).astype(np.uint8)
    spans = np.zeros(w, dtype=np.int32)
    for x in range(w):
        ys = np.where(obj_bin[:, x] > 0)[0]
        if len(ys) > 0:
            spans[x] = int(ys.max() - ys.min() + 1)

    if spans.max() == 0:
        return 0

    k = max(3, int(w * 0.02))
    if k % 2 == 0:
        k += 1
    spans_s = cv2.GaussianBlur(spans.astype(np.float32).reshape(1, -1), (k, 1), 0).flatten()

    n = max(1, int(w * 0.2))
    left_mean = float(spans_s[:n].mean())
    right_mean = float(spans_s[-n:].mean())

    if abs(left_mean - right_mean) < 3.0:
        return 0

    if left_mean > right_mean:
        return +1
    else:
        return -1


def color_uniformity_std(rgba, mask):
    bgr = rgba[:, :, :3]
    inside = mask > 0
    if int(inside.sum()) < 200:
        return 999.0
    pix = bgr[inside].astype(np.float32)
    std_b = float(np.std(pix[:, 0]))
    std_g = float(np.std(pix[:, 1]))
    std_r = float(np.std(pix[:, 2]))
    return (std_b + std_g + std_r) / 3.0


def edge_density(rgba, mask):
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    inside = mask > 0
    denom = float(int(inside.sum()))
    if denom < 200.0:
        return 1.0
    return float(int(edges[inside].sum())) / denom


def score_image(path):
    rgba = read_rgba(path)
    if rgba is None:
        return None

    mask = alpha_mask(rgba)
    if bbox_from_mask(mask) is None:
        return None

    areas = connected_components_areas(mask)
    total_area = int((mask > 0).sum())
    pair_score = 0.0
    if len(areas) >= 2 and total_area > 0:
        a0 = areas[0]
        a1 = areas[1]
        pair_score = float(min(a0, a1)) / float(max(a0, 1))

    # Sole heuristics (kept as is, you can tune later)
    uni = color_uniformity_std(rgba, mask)
    edg = edge_density(rgba, mask)
    hw = height_width_ratio_aligned(mask)
    fill = bbox_fill_ratio(mask)

    sole_score = 0.0
    if uni < 20.0:
        sole_score += 2.0
    if edg < 0.02:
        sole_score += 2.0
    if hw < 0.35:
        sole_score += 3.0
    if fill > 0.70:
        sole_score += 3.0

    diagonal_score = hw
    facing = facing_direction_aligned(mask)

    return {
        "path": path,
        "pair_score": pair_score,
        "sole_score": float(sole_score),
        "diagonal_score": float(diagonal_score),
        "facing": int(facing),
    }


def assign_slots(items):
    unused = items[:]
    slots = {}

    # 1 = pair crossed
    unused.sort(key=lambda d: d["pair_score"], reverse=True)
    if unused and unused[0]["pair_score"] >= 0.20:
        slots[1] = unused.pop(0)

    # 5 = sole
    unused.sort(key=lambda d: d["sole_score"], reverse=True)
    if unused and unused[0]["sole_score"] >= 6.0:
        slots[5] = unused.pop(0)

    # 4 = diagonal
    unused.sort(key=lambda d: d["diagonal_score"], reverse=True)
    if unused and unused[0]["diagonal_score"] >= 0.55:
        slots[4] = unused.pop(0)

    # 2 and 3 = right / left
    right_candidates = [d for d in unused if d["facing"] == +1]
    left_candidates = [d for d in unused if d["facing"] == -1]

    right_candidates.sort(key=lambda d: d["diagonal_score"])
    left_candidates.sort(key=lambda d: d["diagonal_score"])

    if 2 not in slots:
        if right_candidates:
            slots[2] = right_candidates.pop(0)
            if slots[2] in unused:
                unused.remove(slots[2])
        elif unused:
            slots[2] = unused.pop(0)

    if 3 not in slots:
        if left_candidates:
            slots[3] = left_candidates.pop(0)
            if slots[3] in unused:
                unused.remove(slots[3])
        elif unused:
            slots[3] = unused.pop(0)

    # Fill missing slots with remaining images
    for s in [1, 2, 3, 4, 5]:
        if s not in slots and unused:
            slots[s] = unused.pop(0)

    return slots


def make_tmp_name(idx):
    return "__tmp__" + str(idx) + "_" + os.urandom(6).hex() + ".png"


def safe_rename_in_folder(folder, mapping):
    """
    Robust rename:
    - Deletes old __tmp__*.png
    - Renames each selected source file to a unique tmp name
    - Then renames tmp -> final N.png
    - If a file is referenced twice (duplicate), second time is skipped
    """

    # Delete leftover tmp files
    for n in os.listdir(folder):
        if n.startswith("__tmp__") and n.lower().endswith(".png"):
            try:
                os.remove(os.path.join(folder, n))
            except Exception:
                pass

    tmp_map = {}  # final_num -> tmp_path

    for final_num, item in mapping.items():
        old_path = item["path"]

        if not os.path.exists(old_path):
            continue

        tmp_name = make_tmp_name(final_num)
        tmp_path = os.path.join(folder, tmp_name)

        os.rename(old_path, tmp_path)
        tmp_map[final_num] = tmp_path

    for final_num, tmp_path in tmp_map.items():
        final_path = os.path.join(folder, f"{final_num}.png")

        if os.path.exists(final_path):
            try:
                os.remove(final_path)
            except Exception:
                pass

        os.rename(tmp_path, final_path)


def process_hd_folder(hd_path):
    files = []
    for n in os.listdir(hd_path):
        full = os.path.join(hd_path, n)
        if os.path.isfile(full) and n.lower().endswith(".png"):
            files.append(full)

    if not files:
        return

    items = []
    for p in files:
        s = score_image(p)
        if s is not None:
            items.append(s)

    if not items:
        return

    slots = assign_slots(items)

    mapping = {}
    for num, item in slots.items():
        mapping[num] = item

    safe_rename_in_folder(hd_path, mapping)

    print("OK:", hd_path, "renamed:", sorted(mapping.keys()))


def run():
    for current_dir, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(current_dir) == HD_FOLDER_NAME:
            process_hd_folder(current_dir)
            dirs[:] = []
            continue


if __name__ == "__main__":
    run()
