import urllib.request
import json
import os
import datetime

os.makedirs("ai-spike/licenses", exist_ok=True)

urls = {
    "demucs_code": "https://raw.githubusercontent.com/facebookresearch/demucs/main/LICENSE",
    "demucs_readme": "https://raw.githubusercontent.com/facebookresearch/demucs/main/README.md",
    "openunmix_code": "https://raw.githubusercontent.com/sigsep/open-unmix-pytorch/master/LICENSE",
    "openunmix_readme": "https://raw.githubusercontent.com/sigsep/open-unmix-pytorch/master/README.md",
    "musdb18_zenodo": "https://zenodo.org/api/records/1117372",
    "musdb18_hq_zenodo": "https://zenodo.org/api/records/3338373",
    "openunmix_zenodo": "https://zenodo.org/api/records/3370486",
    "openunmix_umxl_zenodo": "https://zenodo.org/api/records/5069689",
    "laion_clap_github_license": "https://raw.githubusercontent.com/LAION-AI/CLAP/main/LICENSE",
    "laion_clap_hf_card": "https://huggingface.co/laion/clap-htsat-unfused/raw/main/README.md",
    "laion_clap_music_hf_card": "https://huggingface.co/lukewys/laion_clap/raw/main/README.md",
    "ms_clap_github_license": "https://raw.githubusercontent.com/microsoft/CLAP/main/LICENSE",
    "ms_clap_github_readme": "https://raw.githubusercontent.com/microsoft/CLAP/main/README.md",
    "minilm_hf_card": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/README.md",
    "uvr_github_license": "https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/LICENSE",
    "kuielab_mdxnet_zenodo": "https://zenodo.org/api/records/5567362",
    "bs_roformer_hf_viperx": "https://huggingface.co/viperx97/tutorial_models/raw/main/README.md",
    "bs_roformer_zfturbo_repo": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/LICENSE"
}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for name, url in urls.items():
    print(f"Fetching {name} from {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            with open(f"ai-spike/licenses/{name}.raw.txt", "w", encoding="utf-8") as f:
                f.write(f"Source URL: {url}\nFetch Date: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n")
                f.write(content)
            print(f"  Successfully fetched {name} ({len(content)} chars)")
    except Exception as e:
        print(f"  Error fetching {name}: {e}")
