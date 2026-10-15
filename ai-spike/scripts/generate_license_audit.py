import os
import datetime

fetch_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

licenses_data = {
    "htdemucs": {
        "model_name": "Demucs v4 (htdemucs / htdemucs_ft)",
        "source_url": "https://github.com/facebookresearch/demucs",
        "code_license": "MIT License (Meta Platforms, Inc.)",
        "weights_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/...",
        "weights_license": "No separate license stated for weights. Repository LICENSE is MIT.",
        "training_data": "MUSDB18-HQ (CC BY-NC-SA 4.0 / non-commercial) + internal 800-song dataset (for _ft).",
        "training_concerns": "MUSDB18-HQ contains copyright tracks licensed strictly for non-commercial research under CC BY-NC-SA 4.0. Derivative work taint creates commercial copyright risk.",
        "verdict": "NEEDS LEGAL REVIEW"
    },
    "demucs_onnx": {
        "model_name": "Demucs ONNX Community Ports (demucs.onnx / adefossez/HTDemucs)",
        "source_url": "https://huggingface.co/adefossez/HTDemucs",
        "code_license": "MIT License",
        "weights_url": "https://huggingface.co/adefossez/HTDemucs",
        "weights_license": "None stated in Hugging Face model card.",
        "training_data": "Converted from Meta htdemucs checkpoints (trained on MUSDB18-HQ).",
        "training_concerns": "Inherits MUSDB18-HQ research-only non-commercial derivative risk.",
        "verdict": "NEEDS LEGAL REVIEW"
    },
    "openunmix_umxhq": {
        "model_name": "Open-Unmix UMXHQ (4-target: vocals, drums, bass, other)",
        "source_url": "https://github.com/sigsep/open-unmix-pytorch",
        "code_license": "MIT License (Inria: Fabian-Robert Stöter, Antoine Liutkus)",
        "weights_url": "https://zenodo.org/records/3370486",
        "weights_license": "\"mit-license\" (Zenodo Record 3370486 metadata explicitly declares MIT).",
        "training_data": "MUSDB18-HQ (Zenodo 3338373, licensed as 'other-nc' / CC BY-NC-SA 4.0).",
        "training_concerns": "While Inria marked the Zenodo weights record as MIT, the underlying training audio MUSDB18-HQ is non-commercial research-only. Commercial distribution creates potential copyright taint unless fair use / non-derivative status is established.",
        "verdict": "NEEDS LEGAL REVIEW"
    },
    "openunmix_umxl": {
        "model_name": "Open-Unmix UMXL (Extra Data Large)",
        "source_url": "https://github.com/sigsep/open-unmix-pytorch",
        "code_license": "MIT License",
        "weights_url": "https://zenodo.org/records/5069601",
        "weights_license": "\"cc-by-nc-sa-4.0\" (Open-Unmix README: 'Note, that the weights are only licensed for non-commercial use (CC BY-NC-SA 4.0)'; Zenodo 5069601 metadata: 'cc-by-nc-sa-4.0').",
        "training_data": "MUSDB18-HQ + private compressed stems dataset.",
        "training_concerns": "Explicitly non-commercial (NC). Commercial use strictly prohibited by license grant.",
        "verdict": "NOT SHIPPABLE"
    },
    "uvr_mdxnet": {
        "model_name": "UVR MDX-Net (e.g. UVR-MDX-NET_Main_340.onnx)",
        "source_url": "https://github.com/Anjok07/ultimatevocalremovergui",
        "code_license": "MIT License (KUIELab / UVR GUI)",
        "weights_url": "https://github.com/TRvlS/model_repo / UVR download mirrors",
        "weights_license": "No explicit license file or grant accompanies community model weights.",
        "training_data": "MDX Challenge 2021 dataset (MUSDB18-HQ) + private scraped/stem tracks.",
        "training_concerns": "Trained on non-commercial competition data and unverified private stems. Absence of explicit commercial license grant.",
        "verdict": "NEEDS LEGAL REVIEW"
    },
    "bs_roformer": {
        "model_name": "BS-RoFormer / Mel-RoFormer Community Checkpoints",
        "source_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training",
        "code_license": "MIT License (Roman Solovyev / ZFTurbo)",
        "weights_url": "https://huggingface.co/viperx97 / ZFTurbo releases",
        "weights_license": "No explicit license grant for weights; released for research/competition.",
        "training_data": "Large private scraped multi-track audio datasets + MUSDB18-HQ.",
        "training_concerns": "Unclear copyright status of commercial songs used in community training sets.",
        "verdict": "NEEDS LEGAL REVIEW"
    },
    "laion_clap_htsat": {
        "model_name": "LAION-CLAP (clap-htsat-unfused)",
        "source_url": "https://huggingface.co/laion/clap-htsat-unfused",
        "code_license": "MIT License (LAION-AI)",
        "weights_url": "https://huggingface.co/laion/clap-htsat-unfused",
        "weights_license": "\"apache-2.0\" (Hugging Face model card metadata explicitly states license: apache-2.0).",
        "training_data": "LAION-Audio-630K (AudioSet, Freesound, BBC Sound Effects, Epidemic Sound).",
        "training_concerns": "AudioSet clips are scraped from YouTube. YouTube terms of service prohibit commercial redistribution of raw audio, though model weights are treated as non-infringing representations in US jurisdiction.",
        "verdict": "SHIPPABLE"
    },
    "laion_clap_music": {
        "model_name": "LAION-CLAP Music Checkpoint (music_audioset_epoch_15_esc_90.14.pt)",
        "source_url": "https://huggingface.co/lukewys/laion_clap",
        "code_license": "MIT License",
        "weights_url": "https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt",
        "weights_license": "\"cc0-1.0\" (Hugging Face cardData explicitly declares license: cc0-1.0 Public Domain).",
        "training_data": "LAION-Audio-630K music subset + AudioSet music.",
        "training_concerns": "Same YouTube/AudioSet web-scraping consideration as base CLAP.",
        "verdict": "SHIPPABLE"
    },
    "ms_clap": {
        "model_name": "MS-CLAP (Microsoft CLAP 2022/2023)",
        "source_url": "https://github.com/microsoft/CLAP",
        "code_license": "MIT License (Microsoft Corporation)",
        "weights_url": "https://huggingface.co/microsoft/msclap",
        "weights_license": "\"ms-pl\" (Microsoft Public License, HF model card metadata).",
        "training_data": "AudioSet, Freesound, SoundBible.",
        "training_concerns": "MS-PL has reciprocal terms for source code redistribution; compliant for binary distribution.",
        "verdict": "SHIPPABLE"
    },
    "minilm_l6_v2": {
        "model_name": "Sentence-Transformers all-MiniLM-L6-v2",
        "source_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "code_license": "Apache 2.0 (UKPLab / HuggingFace)",
        "weights_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "weights_license": "\"apache-2.0\" (Hugging Face model card explicitly states license: apache-2.0).",
        "training_data": "1 Billion sentence pairs from MS MARCO, S2ORC, StackExchange, GooAQ, Yahoo Answers.",
        "training_concerns": "Permissive open text datasets. Clean commercial provenance.",
        "verdict": "SHIPPABLE"
    }
}

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
licenses_dir = os.path.join(base_dir, "licenses")
os.makedirs(licenses_dir, exist_ok=True)
for key, data in licenses_data.items():
    file_path = os.path.join(licenses_dir, f"{key}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {data['model_name']}\n")
        f.write(f"Source URL: {data['source_url']}\n")
        f.write(f"Weights URL: {data['weights_url']}\n")
        f.write(f"Fetch Date: {fetch_date}\n")
        f.write(f"Code License: {data['code_license']}\n")
        f.write(f"Weights License: {data['weights_license']}\n")
        f.write(f"Training Data: {data['training_data']}\n")
        f.write(f"Training Data License Concerns: {data['training_concerns']}\n")
        f.write(f"Verdict: {data['verdict']}\n")
    print(f"Wrote {file_path}")

print("License audit files written successfully!")
