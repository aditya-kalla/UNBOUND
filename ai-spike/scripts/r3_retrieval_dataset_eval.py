import os
import sys
import time
import json
import csv
import io
import urllib.request
import hashlib
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import AutoTokenizer, ClapModel
import torch
import torch.nn.functional as F

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data", "real")
RETRIEVAL_AUDIO_DIR = os.path.join(DATA_DIR, "r3_retrieval_audio")
MANIFEST_FILE = os.path.join(DATA_DIR, "r3_retrieval_manifest.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLAP_DIR = os.path.join(MODELS_DIR, "clap")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RETRIEVAL_AUDIO_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "r3_retrieval.log")
log_out = open(LOG_FILE, "w", encoding="utf-8")

def log(msg):
    print(msg)
    log_out.write(msg + "\n")
    log_out.flush()

def get_sha256(path):
    if not os.path.exists(path):
        return "N/A"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

log("=== R3: PROPER ZERO-SHOT & HYBRID RETRIEVAL EVALUATION ON UNBIASED DATASET ===")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
log("Ground-truth source: MTG-Jamendo Dataset (Bogdanov et al., ISMIR / Machine Learning for Music)")
log("Audio source: Jamendo MP3 30s previews (https://mp3d.jamendo.com/download/track/{id}/mp32/)")

# 1. Fetch MTG-Jamendo Metadata and build 220-track evaluation collection
log("\n--- [PART 1] BUILDING 220-TRACK UNBIASED EVALUATION DATASET ---")

if not os.path.exists(MANIFEST_FILE):
    log("Fetching metadata from MTG-Jamendo GitHub repository...")
    url_genre = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/autotagging_genre.tsv"
    url_mood = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/autotagging_moodtheme.tsv"
    url_inst = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/autotagging_instrument.tsv"
    url_meta = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw.meta.tsv"

    def fetch_tsv_rows(url, max_lines=4000):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            lines = [resp.readline().decode('utf-8') for _ in range(max_lines)]
        reader = csv.reader(io.StringIO(''.join(lines)), delimiter='\t')
        header = next(reader)
        return list(reader)

    log("Downloading tag taxonomy: genre, mood/theme, instruments, track metadata...")
    rows_genre = fetch_tsv_rows(url_genre, 5000)
    rows_mood = fetch_tsv_rows(url_mood, 5000)
    rows_inst = fetch_tsv_rows(url_inst, 5000)
    rows_meta = fetch_tsv_rows(url_meta, 5000)

    # Parse metadata
    meta_dict = {}
    for r in rows_meta:
        if len(r) >= 6:
            meta_dict[r[0]] = {"title": r[3], "artist": r[4], "album": r[5]}

    # Parse tags
    tags_by_id = {}
    genre_by_id = {}
    for r in rows_genre:
        if r:
            tid = r[0]
            if tid not in tags_by_id: tags_by_id[tid] = []
            clean_genres = [t.replace('genre---', '') for t in r[5:] if t]
            tags_by_id[tid].extend(clean_genres)
            if clean_genres and tid not in genre_by_id:
                genre_by_id[tid] = clean_genres[0] # primary ID3 genre

    for r in rows_mood:
        if r:
            tid = r[0]
            if tid not in tags_by_id: tags_by_id[tid] = []
            tags_by_id[tid].extend([t.replace('mood/theme---', '') for t in r[5:] if t])

    for r in rows_inst:
        if r:
            tid = r[0]
            if tid not in tags_by_id: tags_by_id[tid] = []
            tags_by_id[tid].extend([t.replace('instrument---', '') for t in r[5:] if t])

    # Select 220 tracks with rich tags
    selected_tracks = []
    for tid, t_list in tags_by_id.items():
        if tid in meta_dict and tid in genre_by_id and len(t_list) >= 3:
            raw_id = tid.replace("track_", "").lstrip("0")
            if not raw_id: raw_id = "0"
            selected_tracks.append((tid, raw_id, t_list, meta_dict[tid], genre_by_id[tid]))
            if len(selected_tracks) >= 220:
                break

    log(f"Selected {len(selected_tracks)} candidate tracks with multi-dimensional human annotations.")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def download_track_file(item):
        tid, raw_id, t_list, meta, primary_g = item
        audio_dest = os.path.join(RETRIEVAL_AUDIO_DIR, f"{tid}.mp3")
        if os.path.exists(audio_dest) and os.path.getsize(audio_dest) >= 10 * 1024:
            return True
        urls = [
            f"https://mp3d.jamendo.com/download/track/{raw_id}/mp32/",
            f"https://prod-1.storage.jamendo.com/?trackid={raw_id}&format=mp31&from=app-devsite",
            f"https://storage.jamendo.com/download/track/{raw_id}/mp32"
        ]
        for u in urls:
            try:
                req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    with open(audio_dest, "wb") as f:
                        f.write(resp.read())
                if os.path.exists(audio_dest) and os.path.getsize(audio_dest) >= 10 * 1024:
                    return True
            except Exception:
                pass
        return False

    log("Downloading audio files in parallel (16 threads)...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(download_track_file, item) for item in selected_tracks]
        for f in as_completed(futures):
            f.result()

    manifest = []
    for idx, (tid, raw_id, t_list, meta, primary_g) in enumerate(selected_tracks):
        audio_dest = os.path.join(RETRIEVAL_AUDIO_DIR, f"{tid}.mp3")
        if not os.path.exists(audio_dest) or os.path.getsize(audio_dest) < 10 * 1024:
            continue
            
        # Held-out tags: all tags EXCLUDING the primary genre visible to baseline
        held_out = [t.lower() for t in t_list if t.lower() != primary_g.lower()]
        
        manifest.append({
            "track_id": tid,
            "raw_jamendo_id": raw_id,
            "filename": f"{raw_id}.mp3",
            "title": meta["title"],
            "artist": meta["artist"],
            "album": meta["album"],
            "id3_genre": primary_g.lower(),
            "held_out_tags": sorted(list(set(held_out))),
            "all_tags": sorted(list(set([t.lower() for t in t_list]))),
            "audio_path": audio_dest,
            "sha256": get_sha256(audio_dest)
        })
        if (len(manifest)) % 50 == 0:
            log(f"Downloaded & verified {len(manifest)} audio files.")

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
else:
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

log(f"Evaluation collection loaded: {len(manifest)} tracks with human tags.")
log("Sample track visible metadata: filename='{0}', title='{1}', artist='{2}', id3_genre='{3}'".format(
    manifest[0]['filename'], manifest[0]['title'], manifest[0]['artist'], manifest[0]['id3_genre']
))
log(f"Sample track held-out tags (HIDDEN from baseline): {manifest[0]['held_out_tags']}")

# 2. Extract Audio Embeddings using CLAP Audio Tower
log("\n--- [PART 2] COMPUTING REAL CLAP AUDIO EMBEDDINGS (FP32 & INT8) ---")
audio_fp32_sess = ort.InferenceSession(os.path.join(CLAP_DIR, "clap_audio_fp32.onnx"), providers=['CPUExecutionProvider'])
audio_int8_sess = ort.InferenceSession(os.path.join(CLAP_DIR, "clap_audio_int8.onnx"), providers=['CPUExecutionProvider'])

from transformers import AutoProcessor
clap_proc = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

audio_embs_fp32 = []
audio_embs_int8 = []

for idx, item in enumerate(manifest):
    try:
        data, sr = sf.read(item["audio_path"])
        # Take 30s middle excerpt
        dur = len(data) / sr
        if dur > 30.0:
            start_s = (dur - 30.0) / 2.0
            data = data[int(start_s * sr):int((start_s + 30.0) * sr)]
        if data.ndim > 1:
            data = np.mean(data, axis=-1)
        if sr != 48000:
            import scipy.signal
            data = scipy.signal.resample(data, int(len(data) * 48000 / sr))
            sr = 48000
            
        inp = clap_proc(audio=data, sampling_rate=48000, return_tensors="np")
        f_feat = inp["input_features"].astype(np.float32)
        
        # FP32
        out_fp32 = audio_fp32_sess.run(None, {"input_features": f_feat})[0][0]
        out_fp32 = out_fp32 / (np.linalg.norm(out_fp32) + 1e-8)
        audio_embs_fp32.append(out_fp32)
        
        # INT8
        out_int8 = audio_int8_sess.run(None, {"input_features": f_feat})[0][0]
        out_int8 = out_int8 / (np.linalg.norm(out_int8) + 1e-8)
        audio_embs_int8.append(out_int8)
    except Exception as e:
        log(f"Warning on track {item['track_id']}: {e}")
        audio_embs_fp32.append(np.zeros(512, dtype=np.float32))
        audio_embs_int8.append(np.zeros(512, dtype=np.float32))

audio_embs_fp32 = np.array(audio_embs_fp32, dtype=np.float32)
audio_embs_int8 = np.array(audio_embs_int8, dtype=np.float32)
log(f"Computed audio embeddings for all {len(manifest)} tracks: shape {audio_embs_fp32.shape}")

# 3. Define 45 Real Test Queries & Ground Truth from Held-out Tags
log("\n--- [PART 3] DEFINING 45 EVALUATION QUERIES & GROUND TRUTH ---")

queries = [
    # Mood / Atmosphere queries
    "relaxing", "calm", "dark", "energetic", "happy", "melancholic", "epic", "romantic",
    "atmospheric", "chillout", "meditative", "uplifting", "dramatic", "space", "mysterious",
    # Instrument queries
    "synthesizer", "acoustic guitar", "electric guitar", "piano", "violin", "drums",
    "bass", "strings", "saxophone", "flute", "percussion", "keyboard", "horns", "woodwinds",
    # Multi-term vibe combinations
    "relaxing ambient", "dark electronic", "energetic dance", "melancholic piano", "acoustic folk",
    "epic soundtrack", "happy pop", "heavy guitar riff", "chill lounge groove", "space synthesizer",
    # Multilingual & Genre queries
    "soundtrack", "downtempo", "rock", "jazz", "electronic", "techno"
]

tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
text_fp32_sess = ort.InferenceSession(os.path.join(CLAP_DIR, "clap_text_fp32.onnx"), providers=['CPUExecutionProvider'])
text_int8_sess = ort.InferenceSession(os.path.join(CLAP_DIR, "clap_text_int8.onnx"), providers=['CPUExecutionProvider'])

def get_text_emb(q_str, session):
    inp = tokenizer([q_str], padding="max_length", max_length=32, return_tensors="np")
    out = session.run(None, {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"]})[0][0]
    return out / (np.linalg.norm(out) + 1e-8)

# 4. Evaluation Function: Metrics & 95% Bootstrap CI
def dcg_at_k(relevance_array, k=10):
    rel = relevance_array[:k]
    if len(rel) == 0: return 0.0
    discounts = np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(rel / discounts))

def ndcg_at_k(ranked_labels, k=10):
    actual_dcg = dcg_at_k(ranked_labels, k=k)
    ideal_labels = np.sort(ranked_labels)[::-1]
    ideal_dcg = dcg_at_k(ideal_labels, k=k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_retrieval_ranking(ranked_indices, gt_binary):
    # ranked_indices: array of track indices in descending score order
    ranked_gt = gt_binary[ranked_indices]
    
    # Precision@5
    p5 = float(np.mean(ranked_gt[:5]))
    # Recall@20
    total_pos = np.sum(gt_binary)
    r20 = float(np.sum(ranked_gt[:20]) / total_pos) if total_pos > 0 else 0.0
    # nDCG@10
    ndcg10 = ndcg_at_k(ranked_gt, k=10)
    return p5, r20, ndcg10

def bootstrap_ci(metric_list, n_boot=1000, ci=0.95):
    arr = np.array(metric_list)
    means = []
    n = len(arr)
    np.random.seed(42)
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = np.percentile(means, alpha * 100)
    high = np.percentile(means, (1.0 - alpha) * 100)
    mean_val = float(np.mean(arr))
    return mean_val, float(low), float(high)

# 5. Run the 5 Baselines across all Queries
log("\n--- [PART 4] RUNNING RETRIEVAL BASELINES ACROSS ALL QUERIES ---")

raw_eval = {
    "Random": {"p5": [], "r20": [], "ndcg10": []},
    "Metadata_Only": {"p5": [], "r20": [], "ndcg10": []},
    "CLAP_ONNX_FP32": {"p5": [], "r20": [], "ndcg10": []},
    "CLAP_ONNX_INT8": {"p5": [], "r20": [], "ndcg10": []},
    "Hybrid_Meta_Plus_CLAP": {"p5": [], "r20": [], "ndcg10": []}
}

N_TRACKS = len(manifest)

for q_idx, q in enumerate(queries):
    q_tokens = set(q.lower().split())
    
    # Ground truth: defined strictly from held_out_tags (hidden from metadata baseline!)
    gt_binary = np.zeros(N_TRACKS, dtype=np.float32)
    for i, item in enumerate(manifest):
        item_held_out = set(item["held_out_tags"])
        if any(t in item_held_out for t in q_tokens) or any(any(tok in t for tok in q_tokens) for t in item_held_out):
            gt_binary[i] = 1.0
            
    if np.sum(gt_binary) == 0:
        continue # skip query with no positive held-out matches
        
    # 1. Random Baseline
    rnd_idx = np.random.permutation(N_TRACKS)
    p5, r20, n10 = evaluate_retrieval_ranking(rnd_idx, gt_binary)
    raw_eval["Random"]["p5"].append(p5)
    raw_eval["Random"]["r20"].append(r20)
    raw_eval["Random"]["ndcg10"].append(n10)
    
    # 2. Realistic Metadata-Only Baseline
    # Searches ONLY: filename, title, artist, id3_genre (held_out_tags are NOT seen)
    meta_scores = np.zeros(N_TRACKS, dtype=np.float32)
    for i, item in enumerate(manifest):
        vis_str = f"{item['filename']} {item['title']} {item['artist']} {item['id3_genre']}".lower()
        vis_tokens = set(vis_str.split())
        match_count = len(q_tokens.intersection(vis_tokens))
        # Exact substring boost
        if q.lower() in vis_str: match_count += 2.0
        meta_scores[i] = match_count
    meta_idx = np.argsort(meta_scores)[::-1]
    p5, r20, n10 = evaluate_retrieval_ranking(meta_idx, gt_binary)
    raw_eval["Metadata_Only"]["p5"].append(p5)
    raw_eval["Metadata_Only"]["r20"].append(r20)
    raw_eval["Metadata_Only"]["ndcg10"].append(n10)
    
    # 3. CLAP ONNX FP32
    q_emb_fp32 = get_text_emb(q, text_fp32_sess)
    cos_fp32 = np.dot(audio_embs_fp32, q_emb_fp32)
    clap_fp32_idx = np.argsort(cos_fp32)[::-1]
    p5, r20, n10 = evaluate_retrieval_ranking(clap_fp32_idx, gt_binary)
    raw_eval["CLAP_ONNX_FP32"]["p5"].append(p5)
    raw_eval["CLAP_ONNX_FP32"]["r20"].append(r20)
    raw_eval["CLAP_ONNX_FP32"]["ndcg10"].append(n10)
    
    # 4. CLAP ONNX INT8
    q_emb_int8 = get_text_emb(q, text_int8_sess)
    cos_int8 = np.dot(audio_embs_int8, q_emb_int8)
    clap_int8_idx = np.argsort(cos_int8)[::-1]
    p5, r20, n10 = evaluate_retrieval_ranking(clap_int8_idx, gt_binary)
    raw_eval["CLAP_ONNX_INT8"]["p5"].append(p5)
    raw_eval["CLAP_ONNX_INT8"]["r20"].append(r20)
    raw_eval["CLAP_ONNX_INT8"]["ndcg10"].append(n10)
    
    # 5. Hybrid Baseline (Metadata Keyword Filter + CLAP Semantic Re-rank)
    # Normalize scores to [0, 1]
    norm_meta = (meta_scores - np.min(meta_scores)) / (np.max(meta_scores) - np.min(meta_scores) + 1e-5)
    norm_clap = (cos_fp32 - np.min(cos_fp32)) / (np.max(cos_fp32) - np.min(cos_fp32) + 1e-5)
    hybrid_score = 0.5 * norm_meta + 0.5 * norm_clap
    hybrid_idx = np.argsort(hybrid_score)[::-1]
    p5, r20, n10 = evaluate_retrieval_ranking(hybrid_idx, gt_binary)
    raw_eval["Hybrid_Meta_Plus_CLAP"]["p5"].append(p5)
    raw_eval["Hybrid_Meta_Plus_CLAP"]["r20"].append(r20)
    raw_eval["Hybrid_Meta_Plus_CLAP"]["ndcg10"].append(n10)

log("\n--- RETRIEVAL BENCHMARK RESULTS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS (MEASURED) ---")
log("| Baseline / Model | Precision@5 (95% CI) | Recall@20 (95% CI) | nDCG@10 (95% CI) |")
log("|---|---|---|---|")

for name, metrics in raw_eval.items():
    p5_m, p5_l, p5_h = bootstrap_ci(metrics["p5"])
    r20_m, r20_l, r20_h = bootstrap_ci(metrics["r20"])
    n10_m, n10_l, n10_h = bootstrap_ci(metrics["ndcg10"])
    log(f"| MEASURED: {name} | **{p5_m:.4f}** [{p5_l:.4f}, {p5_h:.4f}] | **{r20_m:.4f}** [{r20_l:.4f}, {r20_h:.4f}] | **{n10_m:.4f}** [{n10_l:.4f}, {n10_h:.4f}] |")

# 6. Audio-to-Audio "More Like This" Evaluation
log("\n--- [PART 5] AUDIO-TO-AUDIO 'MORE LIKE THIS' EVALUATION ---")
log("Relevance criterion: Jaccard overlap of full ground-truth tag annotations >= 0.20")

a2a_p5 = []
a2a_r20 = []
a2a_ndcg10 = []

for i in range(N_TRACKS):
    tags_i = set(manifest[i]["all_tags"])
    # Binary ground truth relevance with all other tracks
    gt_a2a = np.zeros(N_TRACKS, dtype=np.float32)
    for j in range(N_TRACKS):
        if i == j: continue
        tags_j = set(manifest[j]["all_tags"])
        jaccard = len(tags_i.intersection(tags_j)) / (len(tags_i.union(tags_j)) + 1e-8)
        if jaccard >= 0.20:
            gt_a2a[j] = 1.0
            
    if np.sum(gt_a2a) == 0:
        continue
        
    # Audio embedding cosine similarity
    sims = np.dot(audio_embs_fp32, audio_embs_fp32[i])
    sims[i] = -1e9 # exclude self
    ranked_j = np.argsort(sims)[::-1]
    
    p5, r20, n10 = evaluate_retrieval_ranking(ranked_j, gt_a2a)
    a2a_p5.append(p5)
    a2a_r20.append(r20)
    a2a_ndcg10.append(n10)

a2a_p5_m, a2a_p5_l, a2a_p5_h = bootstrap_ci(a2a_p5)
a2a_r20_m, a2a_r20_l, a2a_r20_h = bootstrap_ci(a2a_r20)
a2a_ndcg10_m, a2a_ndcg10_l, a2a_ndcg10_h = bootstrap_ci(a2a_ndcg10)

log(f"MEASURED: Audio-to-Audio Precision@5:  **{a2a_p5_m:.4f}** (95% CI: [{a2a_p5_l:.4f}, {a2a_p5_h:.4f}])")
log(f"MEASURED: Audio-to-Audio Recall@20:    **{a2a_r20_m:.4f}** (95% CI: [{a2a_r20_l:.4f}, {a2a_r20_h:.4f}])")
log(f"MEASURED: Audio-to-Audio nDCG@10:      **{a2a_ndcg10_m:.4f}** (95% CI: [{a2a_ndcg10_l:.4f}, {a2a_ndcg10_h:.4f}])")

log("\n=== R3 EVALUATION COMPLETED ===")
log_out.close()
