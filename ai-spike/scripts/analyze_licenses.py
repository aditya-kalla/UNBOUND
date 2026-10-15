import json
import glob
import os

print("--- EXAMINING FETCHED LICENSES ---")

# 1. Demucs
with open("ai-spike/licenses/demucs_code.raw.txt", encoding="utf-8") as f:
    demucs_code = f.read()
with open("ai-spike/licenses/demucs_readme.raw.txt", encoding="utf-8") as f:
    demucs_readme = f.read()

print("\n[DEMUCS CODE LICENSE]:")
print(demucs_code.split("\n\n", 1)[1][:300])

print("\n[DEMUCS README LICENSE SECTION]:")
if "License" in demucs_readme:
    idx = demucs_readme.find("License")
    print(demucs_readme[idx:idx+600])
else:
    print("No License heading in README")

# 2. Open-Unmix
with open("ai-spike/licenses/openunmix_code.raw.txt", encoding="utf-8") as f:
    umx_code = f.read()
with open("ai-spike/licenses/openunmix_readme.raw.txt", encoding="utf-8") as f:
    umx_readme = f.read()

print("\n[OPEN-UNMIX CODE LICENSE]:")
print(umx_code.split("\n\n", 1)[1][:300])

print("\n[OPEN-UNMIX README LICENSE SECTION]:")
if "License" in umx_readme:
    idx = umx_readme.find("License")
    print(umx_readme[idx:idx+600])

# 3. MUSDB18 Zenodo
with open("ai-spike/licenses/musdb18_zenodo.raw.txt", encoding="utf-8") as f:
    musdb_data = json.loads(f.read().split("\n\n", 1)[1])
    print("\n[MUSDB18 ZENODO LICENSE]:", musdb_data.get("metadata", {}).get("license"))

with open("ai-spike/licenses/musdb18_hq_zenodo.raw.txt", encoding="utf-8") as f:
    musdb_hq_data = json.loads(f.read().split("\n\n", 1)[1])
    print("[MUSDB18-HQ ZENODO LICENSE]:", musdb_hq_data.get("metadata", {}).get("license"))

# 4. Open-Unmix Zenodo
with open("ai-spike/licenses/openunmix_zenodo.raw.txt", encoding="utf-8") as f:
    umx_zenodo = json.loads(f.read().split("\n\n", 1)[1])
    print("[OPEN-UNMIX UMXHQ ZENODO LICENSE]:", umx_zenodo.get("metadata", {}).get("license"))

with open("ai-spike/licenses/openunmix_umxl_zenodo.raw.txt", encoding="utf-8") as f:
    umxl_zenodo = json.loads(f.read().split("\n\n", 1)[1])
    print("[OPEN-UNMIX UMXL ZENODO LICENSE]:", umxl_zenodo.get("metadata", {}).get("license"))

# 5. KUIELab MDX-Net Zenodo
with open("ai-spike/licenses/kuielab_mdxnet_zenodo.raw.txt", encoding="utf-8") as f:
    mdx_zenodo = json.loads(f.read().split("\n\n", 1)[1])
    print("[KUIELAB MDX-NET ZENODO LICENSE]:", mdx_zenodo.get("metadata", {}).get("license"))

# 6. LAION-CLAP HF
with open("ai-spike/licenses/laion_clap_hf_card.raw.txt", encoding="utf-8") as f:
    clap_hf = f.read()
    print("\n[LAION-CLAP HF CARD HEADER]:")
    print(clap_hf.split("\n\n", 1)[1][:400])

# 7. MS-CLAP
with open("ai-spike/licenses/ms_clap_github_license.raw.txt", encoding="utf-8") as f:
    ms_clap = f.read()
    print("\n[MS-CLAP LICENSE]:")
    print(ms_clap.split("\n\n", 1)[1][:400])

# 8. MiniLM
with open("ai-spike/licenses/minilm_hf_card.raw.txt", encoding="utf-8") as f:
    minilm_hf = f.read()
    print("\n[MINILM HF CARD HEADER]:")
    print(minilm_hf.split("\n\n", 1)[1][:400])

