from PIL import Image
import os

# =========================
# CONFIG
# =========================

ROOT_DIR = r"C:\Users\table\Documents\Carlos\Personales\B\Zapatos\Zapatillas Ninos"

OUTPUT_WIDTH = 1500
OUTPUT_HEIGHT = 1500
MARGIN = 200
ALPHA_THRESHOLD = 1

HD_FOLDER_NAME = "HD"


# =========================
# IMAGE HELPERS
# =========================

def bbox_from_alpha(img, alpha_threshold):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.getchannel("A")
    mask = alpha.point(lambda a: 255 if a > alpha_threshold else 0)
    return mask.getbbox()


def contain_size(src_w, src_h, max_w, max_h):
    scale = min(max_w / src_w, max_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    return max(1, new_w), max(1, new_h)


def has_transparency(img):
    """
    True if image has alpha channel AND there is at least one transparent pixel.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.getchannel("A")
    mn, mx = alpha.getextrema()  # (min_alpha, max_alpha)
    return mn < 255


def process_one_png(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")

    bbox = bbox_from_alpha(img, ALPHA_THRESHOLD)
    if bbox is None:
        # No visible content
        return False

    obj = img.crop(bbox)
    obj_w, obj_h = obj.size

    max_w = max(1, OUTPUT_WIDTH - 2 * MARGIN)
    max_h = max(1, OUTPUT_HEIGHT - 2 * MARGIN)

    new_w, new_h = contain_size(obj_w, obj_h, max_w, max_h)
    obj_resized = obj.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
    x = (OUTPUT_WIDTH - new_w) // 2
    y = (OUTPUT_HEIGHT - new_h) // 2
    canvas.paste(obj_resized, (x, y), obj_resized)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    return True


# =========================
# DIRECTORY LOGIC
# =========================

def list_pngs_in_dir(folder):
    out = []
    try:
        for name in os.listdir(folder):
            if name.lower().endswith(".png"):
                full = os.path.join(folder, name)
                if os.path.isfile(full):
                    out.append(full)
    except OSError:
        pass
    return out


def process_directory(folder):
    """
    Rules:
    - If this folder contains a subfolder named "HD", skip this folder.
    - If no "HD" and there are .png files in this folder:
        - Create "HD" folder
        - For each .png in this folder:
            - If it has transparency, process and save to HD with same name
    """
    hd_path = os.path.join(folder, HD_FOLDER_NAME)

    # If HD exists, skip this folder (as requested)
    if os.path.isdir(hd_path):
        return

    pngs = list_pngs_in_dir(folder)
    if not pngs:
        return

    # Filter only png with transparency
    transparent_pngs = []
    for p in pngs:
        try:
            img = Image.open(p)
            if has_transparency(img):
                transparent_pngs.append(p)
        except Exception:
            # ignore unreadable files
            continue

    if not transparent_pngs:
        return

    os.makedirs(hd_path, exist_ok=True)

    for p in transparent_pngs:
        out_path = os.path.join(hd_path, os.path.basename(p))
        try:
            ok = process_one_png(p, out_path)
            if ok:
                print("OK  :", p, "->", out_path)
            else:
                print("SKIP:", p, "(no visible content)")
        except Exception as e:
            print("ERR :", p, "->", str(e))


def run():
    if not os.path.isdir(ROOT_DIR):
        raise ValueError("ROOT_DIR does not exist: " + ROOT_DIR)

    for current_dir, dirs, files in os.walk(ROOT_DIR):
        # Do not enter any HD folder
        dirs[:] = [d for d in dirs if d != HD_FOLDER_NAME]

        # Process current directory according to the rules
        process_directory(current_dir)


if __name__ == "__main__":
    run()
