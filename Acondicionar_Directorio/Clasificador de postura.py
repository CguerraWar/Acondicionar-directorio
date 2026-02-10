import os
import cv2
import numpy as np

ROOT_DIR = r"C:\Users\table\Documents\Carlos\Personales\B\Zapatos\Ballerinas\Listas\281"
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
        # no alpha, create full alpha
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


def connected_components_areas(mask):
    # mask is 0/255
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


def white_ratio(rgba, mask):
    # compute ratio of near-white pixels inside object
    bgr = rgba[:, :, :3]
    inside = mask > 0
    if inside.sum() < 200:
        return 0.0
    pix = bgr[inside]
    # near-white in BGR
    w = np.mean((pix[:, 0] > 200) & (pix[:, 1] > 200) & (pix[:, 2] > 200))
    return float(w)


def height_width_ratio_aligned(mask):
    # align by PCA so major axis is horizontal
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
    # returns +1 = facing right, -1 = facing left, 0 = unknown
    # steps:
    # 1) align by PCA to horizontal
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

    # 2) build thickness profile along x (vertical span for each column)
    obj_bin = (obj > 0).astype(np.uint8)
    spans = np.zeros(w, dtype=np.int32)
    for x in range(w):
        ys = np.where(obj_bin[:, x] > 0)[0]
        if len(ys) > 0:
            spans[x] = int(ys.max() - ys.min() + 1)

    if spans.max() == 0:
        return 0

    # smooth
    k = max(3, int(w * 0.02))
    if k % 2 == 0:
        k += 1
    spans_s = cv2.GaussianBlur(spans.astype(np.float32).reshape(1, -1), (k, 1), 0).flatten()

    # 3) compare average thickness in first/last 20%
    n = max(1, int(w * 0.2))
    left_mean = float(spans_s[:n].mean())
    right_mean = float(spans_s[-n:].mean())

    # heuristic:
    # toe area often has lower height than heel collar (heel higher),
    # so the thicker end tends to be heel side.
    # if left thicker => heel on left => facing right
    # if right thicker => heel on right => facing left
    if abs(left_mean - right_mean) < 3.0:
        return 0

    if left_mean > right_mean:
        return +1
    else:
        return -1


def score_image(path):
    rgba = read_rgba(path)
    if rgba is None:
        return None

    mask = alpha_mask(rgba)
    bb = bbox_from_mask(mask)
    if bb is None:
        return None

    # pair score from connected components
    areas = connected_components_areas(mask)
    total_area = int((mask > 0).sum())
    pair_score = 0.0
    if len(areas) >= 2 and total_area > 0:
        # if 2 biggest components are both significant
        a0 = areas[0]
        a1 = areas[1]
        pair_score = float(min(a0, a1)) / float(max(a0, 1))

    # sole score from white ratio
    sole_score = white_ratio(rgba, mask)

    # diagonal score from aligned h/w ratio
    hw = height_width_ratio_aligned(mask)
    diagonal_score = hw  # higher -> more diagonal/vertical volume

    # facing
    facing = facing_direction_aligned(mask)

    return {
        "path": path,
        "pair_score": pair_score,
        "sole_score": sole_score,
        "diagonal_score": diagonal_score,
        "facing": facing,
    }


def assign_slots(items):
    # items: list of dicts
    unused = items[:]
    slots = {}

    # slot 1: best pair
    unused.sort(key=lambda d: d["pair_score"], reverse=True)
    if unused and unused[0]["pair_score"] >= 0.20:
        slots[1] = unused.pop(0)

    # slot 5: best sole
    unused.sort(key=lambda d: d["sole_score"], reverse=True)
    if unused and unused[0]["sole_score"] >= 0.20:
        slots[5] = unused.pop(0)

    # slot 4: best diagonal among remaining
    # typical side views have smaller h/w than diagonal; tune threshold if needed
    unused.sort(key=lambda d: d["diagonal_score"], reverse=True)
    if unused and unused[0]["diagonal_score"] >= 0.55:
        slots[4] = unused.pop(0)

    # remaining: try to pick right/left by facing direction
    right_candidates = [d for d in unused if d["facing"] == +1]
    left_candidates = [d for d in unused if d["facing"] == -1]
    unknown_candidates = [d for d in unused if d["facing"] == 0]

    # choose strongest by diagonal_score lower (more side-like) for 2 and 3
    right_candidates.sort(key=lambda d: d["diagonal_score"])
    left_candidates.sort(key=lambda d: d["diagonal_score"])
    unknown_candidates.sort(key=lambda d: d["diagonal_score"])

    if 2 not in slots:
        if right_candidates:
            slots[2] = right_candidates.pop(0)
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

    # fill remaining slots in order 1..5 if missing and we still have images
    for s in [1, 2, 3, 4, 5]:
        if s not in slots and unused:
            slots[s] = unused.pop(0)

    return slots


def safe_rename_in_folder(folder, mapping):
    # mapping: {final_number: path}
    # two-phase rename to avoid collisions
    temp_paths = []

    for num, item in mapping.items():
        old_path = item["path"]
        tmp_name = f"__tmp__{num}__.png"
        tmp_path = os.path.join(folder, tmp_name)

        j = 1
        while os.path.exists(tmp_path):
            tmp_name = f"__tmp__{num}__{j}__.png"
            tmp_path = os.path.join(folder, tmp_name)
            j += 1

        os.rename(old_path, tmp_path)
        temp_paths.append((num, tmp_path))

    for num, tmp_path in temp_paths:
        final_path = os.path.join(folder, f"{num}.png")
        os.rename(tmp_path, final_path)


def process_hd_folder(hd_path):
    names = []
    for n in os.listdir(hd_path):
        full = os.path.join(hd_path, n)
        if os.path.isfile(full) and n.lower().endswith(".png"):
            names.append(n)

    if not names:
        return

    # score all images
    items = []
    for n in names:
        full = os.path.join(hd_path, n)
        s = score_image(full)
        if s is not None:
            items.append(s)

    if not items:
        return

    slots = assign_slots(items)

    # build mapping and rename
    mapping = {}
    for num, item in slots.items():
        mapping[num] = item

    # only rename slots that exist
    # also, do not require all 1..5 to be present
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



