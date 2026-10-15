import glob
import hashlib
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(logs_dir, exist_ok=True)
    models = sorted(glob.glob(os.path.join(models_dir, "*")))
    lines = []
    header = f"{'Filename':<35} {'Size (MB)':<12} {'SHA256'}"
    lines.append(header)
    lines.append("-" * 115)
    print(header)
    print("-" * 115)
    for p in models:
        if os.path.isdir(p):
            continue
        size_mb = os.path.getsize(p) / (1024 * 1024)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        line = f"{os.path.basename(p):<35} {size_mb:<12.2f} {h.hexdigest()}"
        lines.append(line)
        print(line)

    with open(os.path.join(logs_dir, "model_hashes.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
