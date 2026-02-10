import os

# =========================
# CONFIG
# =========================

ROOT_DIR = r"C:\Users\table\Documents\Carlos\Personales\B\Zapatos"
HD_FOLDER_NAME = "HD"


# =========================
# HELPERS
# =========================

def is_numeric_stem(filename):
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".png":
        return False
    return stem.isdigit()


def sort_key_for_png(filename):
    """
    Sorting rules:
    1) Numeric stems first, ordered by integer value (ascending)
    2) Non-numeric stems next, ordered by filename (case-insensitive)
    """
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".png":
        # Put non-png at the end, but we will ignore them anyway
        return (2, 0, filename.lower())

    if stem.isdigit():
        try:
            return (0, int(stem), filename.lower())
        except Exception:
            # Extremely large numbers are still fine in Python, but just in case
            return (0, 0, filename.lower())
    else:
        return (1, 0, filename.lower())


def rename_pngs_in_hd(hd_path):
    files = []
    try:
        for name in os.listdir(hd_path):
            full = os.path.join(hd_path, name)
            if os.path.isfile(full) and name.lower().endswith(".png"):
                files.append(name)
    except OSError:
        return

    if not files:
        return

    # Sort by the rule requested
    files_sorted = sorted(files, key=sort_key_for_png)

    # Two-phase rename to avoid collisions:
    # First rename to temporary names, then rename to final 1.png..N.png
    temp_names = []
    for i, old_name in enumerate(files_sorted, start=1):
        old_path = os.path.join(hd_path, old_name)
        tmp_name = f"__tmp__{i}__.png"
        tmp_path = os.path.join(hd_path, tmp_name)

        # Ensure tmp name does not exist
        j = 1
        while os.path.exists(tmp_path):
            tmp_name = f"__tmp__{i}__{j}__.png"
            tmp_path = os.path.join(hd_path, tmp_name)
            j += 1

        os.rename(old_path, tmp_path)
        temp_names.append(tmp_name)

    # Now rename temp -> final 1.png..N.png
    for idx, tmp_name in enumerate(temp_names, start=1):
        tmp_path = os.path.join(hd_path, tmp_name)
        final_name = f"{idx}.png"
        final_path = os.path.join(hd_path, final_name)

        # If final exists (should not after temp rename), pick next available
        if os.path.exists(final_path):
            k = idx
            while os.path.exists(os.path.join(hd_path, f"{k}.png")):
                k += 1
            final_name = f"{k}.png"
            final_path = os.path.join(hd_path, final_name)

        os.rename(tmp_path, final_path)

    print("Renamed in:", hd_path, "Count:", len(files_sorted))


def run():
    if not os.path.isdir(ROOT_DIR):
        raise ValueError("ROOT_DIR does not exist: " + ROOT_DIR)

    for current_dir, dirs, files in os.walk(ROOT_DIR):
        # If this directory itself is HD, process it
        if os.path.basename(current_dir) == HD_FOLDER_NAME:
            rename_pngs_in_hd(current_dir)
            # No need to walk inside further (but you can allow it)
            dirs[:] = []
            continue

        # Otherwise keep walking
        # (We do not block entering HD because we want to find/process them)
        # Nothing else to do here

if __name__ == "__main__":
    run()
