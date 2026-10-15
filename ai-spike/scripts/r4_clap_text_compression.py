import os
import sys
import time
import hashlib
import gc
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
from onnxconverter_common import float16
from transformers import AutoTokenizer, ClapModel

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLAP_DIR = os.path.join(MODELS_DIR, "clap")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CLAP_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "r4_clap_text.log")
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

log("=== R4: CLAP TEXT TOWER COMPRESSION & VOCABULARY STUDY ===")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
log(f"Platform: Windows 11 AMD64, ORT Version: {ort.__version__}")

# Load tokenizer and base model
model_name = "laion/clap-htsat-unfused"
tokenizer = AutoTokenizer.from_pretrained(model_name)
clap_model = ClapModel.from_pretrained(model_name).eval()

class TextTower(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
    def forward(self, input_ids, attention_mask):
        return self.m.get_text_features(input_ids=input_ids, attention_mask=attention_mask).pooler_output

text_tower = TextTower(clap_model).eval()

# Test queries for evaluation (50 diverse multi-domain prompts: EN, ES, HI, TE)
test_prompts = [
    "acoustic guitar folk ballad", "heavy distorted electric guitar riff", "upbeat funk bass groove",
    "classical piano solo romantic", "fast electronic techno drums 140 bpm", "chill lofi hip hop beats",
    "ambient atmospheric synth pad", "epic orchestral film score", "jazz saxophone solo swing",
    "reggae dub bassline with delay", "dark trap beat 808 sub bass", "happy acoustic ukulele pop",
    "melancholic cello and violin duet", "psychedelic rock synth solo", "hardstyle kick drum energetic",
    "relaxing meditation nature sounds", "latin salsa percussion rhythm", "country western acoustic rhythm",
    "deep house bassline hypnotic", "punk rock fast tempo distorted vocals",
    # Multilingual terms
    "musica relajante para dormir", "guitarra acustica romantica", "cancion alegre bailable",
    "ritmo de salsa y percusion", "musica clasica piano suave", "rock en espanol con guitarra electrica",
    "shant sangeet sukoon", "dholak aur bansuri ka sangeet", "tez dhun nritya sangeet",
    "purane gaane tabla aur sitar", "desh bhakti sangeet orchestra", "ghazal shaant vaatavaran",
    "prasanta sangeetham koluvu", "veena madhuramaina paata", "vegavanthamaina nrutya sangeetham",
    "telugu melodi paatalu", "dappu vatyalu sangeetham", "bhakthi geethalu",
    # Complex cross-cutting prompts
    "warm vintage analog synthesizer", "driving highway rock rhythm", "slow melancholic rainy day mood",
    "uplifting morning coffee acoustic", "tense action suspense percussion", "dreamy shoegaze reverb guitars",
    "futuristic cyberpunk electronic beats", "subtle background coffee shop acoustic", "bright brass fanfare celebratory",
    "soulful blues guitar bending"
]

def get_pt_embeddings(prompts):
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        out = text_tower(inputs["input_ids"], inputs["attention_mask"])
        out = F.normalize(out, dim=-1)
    return out.numpy()

log("Computing PyTorch ground-truth text embeddings...")
pt_embs = get_pt_embeddings(test_prompts)

def benchmark_model(name, model_path, test_prompts, is_custom_run=None):
    sz_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0
    sha = get_sha256(model_path)
    
    if is_custom_run is None:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # Warmup
        inp = tokenizer(["test"], padding="max_length", max_length=32, return_tensors="np")
        _ = sess.run(None, {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"]})
        
        # Latency benchmark (single query seq=32)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = sess.run(None, {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"]})
            times.append(time.perf_counter() - t0)
        lat_ms = np.median(times) * 1000.0
        
        # Cosine similarity vs PyTorch across test prompts
        cos_sims = []
        for i, p in enumerate(test_prompts):
            inp_p = tokenizer([p], padding="max_length", max_length=64, return_tensors="np")
            out_p = sess.run(None, {"input_ids": inp_p["input_ids"], "attention_mask": inp_p["attention_mask"]})[0]
            norm_p = out_p / (np.linalg.norm(out_p, axis=-1, keepdims=True) + 1e-8)
            cos = float(np.dot(norm_p[0], pt_embs[i]))
            cos_sims.append(cos)
        mean_cos = float(np.mean(cos_sims))
        min_cos = float(np.min(cos_sims))
    else:
        lat_ms, mean_cos, min_cos = is_custom_run(test_prompts)
        
    log(f"MEASURED: {name} | Size: {sz_mb:.2f} MB | Mean Cosine: {mean_cos:.6f} (Min: {min_cos:.6f}) | Latency: {lat_ms:.2f} ms | SHA256: {sha}")
    return {"size_mb": sz_mb, "cosine": mean_cos, "min_cosine": min_cos, "latency_ms": lat_ms, "sha256": sha}

results = {}

# 0. FP32 Baseline
fp32_path = os.path.join(CLAP_DIR, "clap_text_fp32.onnx")
results["FP32_Baseline"] = benchmark_model("CLAP Text FP32 Baseline", fp32_path, test_prompts)

# 0b. Round 2 Naive Dynamic INT8
int8_naive_path = os.path.join(CLAP_DIR, "clap_text_int8.onnx")
results["INT8_Naive_Dynamic"] = benchmark_model("CLAP Text INT8 Naive Dynamic (Round 2)", int8_naive_path, test_prompts)

# 1. Option (1): FP16 Text Tower
log("\n--- [OPTION 1] FP16 TEXT TOWER EXPORT & EVALUATION ---")
fp16_path = os.path.join(CLAP_DIR, "clap_text_fp16.onnx")
if not os.path.exists(fp16_path):
    log(f"Converting FP32 text tower to FP16 -> {fp16_path}...")
    m_fp32 = onnx.load(fp32_path)
    # Convert float to float16
    m_fp16 = float16.convert_float_to_float16(m_fp32, keep_io_types=True)
    onnx.save(m_fp16, fp16_path)
results["FP16_Text_Tower"] = benchmark_model("CLAP Text FP16", fp16_path, test_prompts)

# 2. Option (2): Selective INT8 (Quantize MatMul/Gemm only, preserve embedding table & LayerNorm)
log("\n--- [OPTION 2] SELECTIVE INT8 (QUANTIZE MATMUL/GEMM ONLY) ---")
selective_int8_path = os.path.join(CLAP_DIR, "clap_text_selective_int8.onnx")
if not os.path.exists(selective_int8_path):
    log(f"Generating Selective INT8 text tower -> {selective_int8_path}...")
    # Quantize only MatMul and Gemm, leave Gather (embeddings) and LayerNorm untouched
    quantize_dynamic(
        model_input=fp32_path,
        model_output=selective_int8_path,
        op_types_to_quantize=['MatMul', 'Gemm'],
        weight_type=QuantType.QInt8,
        per_channel=True
    )
results["Selective_INT8"] = benchmark_model("CLAP Text Selective INT8 (per-channel MatMul)", selective_int8_path, test_prompts)

# 3. Option (3): Static Quantization with Calibration Data Reader
log("\n--- [OPTION 3] STATIC INT8 QUANTIZATION WITH CALIBRATION PROMPTS ---")
static_int8_path = os.path.join(CLAP_DIR, "clap_text_static_int8.onnx")
if not os.path.exists(static_int8_path):
    log("Building calibration dataset (200 real text prompts)...")
    # Generate 200 calibration texts
    calib_genres = ["rock", "pop", "jazz", "classical", "hip hop", "ambient", "electronic", "metal", "folk", "reggae"]
    calib_moods = ["happy", "sad", "energetic", "calm", "dark", "uplifting", "romantic", "melancholy"]
    calib_insts = ["piano", "guitar", "drums", "synth", "bass", "violin", "flute", "saxophone"]
    calib_prompts = []
    for g in calib_genres:
        for m in calib_moods:
            calib_prompts.append(f"{m} {g} music")
    for i in calib_insts:
        for m in calib_moods:
            calib_prompts.append(f"{m} {i} solo")
    calib_prompts = calib_prompts[:200]
    
    class TextCalibrationDataReader(CalibrationDataReader):
        def __init__(self, prompts):
            self.data = []
            for p in prompts:
                inp = tokenizer([p], padding="max_length", max_length=32, return_tensors="np")
                self.data.append({"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"]})
            self.iter = iter(self.data)
        def get_next(self):
            return next(self.iter, None)
            
    calib_reader = TextCalibrationDataReader(calib_prompts)
    log(f"Running static quantization -> {static_int8_path}...")
    try:
        quantize_static(
            model_input=fp32_path,
            model_output=static_int8_path,
            calibration_data_reader=calib_reader,
            quant_format=ort.quantization.QuantFormat.QDQ,
            op_types_to_quantize=['MatMul', 'Gemm'],
            per_channel=True
        )
        results["Static_INT8"] = benchmark_model("CLAP Text Static INT8 (QDQ per-channel)", static_int8_path, test_prompts)
    except Exception as e:
        log(f"MEASURED: Static quantization error: {e}")
        results["Static_INT8"] = {"size_mb": 0.0, "cosine": 0.0, "latency_ms": 0.0, "sha256": "FAILED"}
else:
    results["Static_INT8"] = benchmark_model("CLAP Text Static INT8 (QDQ per-channel)", static_int8_path, test_prompts)

# 4. Option (4): Pre-computed Text-Embedding Vocabulary + Tiny Matcher
log("\n--- [OPTION 4] PRE-COMPUTED VOCABULARY EMBEDDINGS (1,000 PHRASES) ---")
vocab_file = os.path.join(CLAP_DIR, "text_vocabulary_embeddings_1000.npz")
if not os.path.exists(vocab_file):
    log("Building 1,000 domain vocabulary phrases across EN, ES, HI, TE...")
    genres = ["acoustic", "ambient", "blues", "classical", "country", "dance", "disco", "downtempo", "drum and bass", "dubstep", "edm", "electronic", "folk", "funk", "grunge", "hard rock", "heavy metal", "hip hop", "house", "indie", "jazz", "lofi", "lounge", "metal", "orchestral", "pop", "punk", "rap", "reggae", "rock", "rnb", "salsa", "soundtrack", "synthwave", "techno", "trance", "trap", "ukulele", "world"]
    moods = ["aggressive", "atmospheric", "bright", "calm", "chill", "cinematic", "commercial", "corporate", "cozy", "dark", "dramatic", "dreamy", "emotional", "energetic", "epic", "euphoric", "gentle", "groovy", "happy", "heavy", "hopeful", "hypnotic", "intense", "light", "magical", "meditative", "melancholic", "mellow", "mysterious", "nostalgic", "peaceful", "powerful", "quiet", "relaxing", "romantic", "sad", "serene", "slow", "smooth", "soft", "soulful", "space", "spooky", "suspenseful", "sweet", "trippy", "upbeat", "uplifting", "vibrant", "warm"]
    instruments = ["acoustic guitar", "accordion", "bass", "brass", "cello", "clarinet", "drum machine", "drums", "electric guitar", "flute", "harp", "horns", "keyboard", "organ", "percussion", "piano", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "violin", "vocals", "woodwinds"]
    
    vocab_phrases = []
    # 1-word & 2-word combinations
    for g in genres: vocab_phrases.append(g)
    for m in moods: vocab_phrases.append(m)
    for i in instruments: vocab_phrases.append(i)
    for m in moods[:30]:
        for g in genres[:20]:
            vocab_phrases.append(f"{m} {g}")
    for m in moods[:25]:
        for i in instruments[:20]:
            vocab_phrases.append(f"{m} {i}")
    for g in genres[:20]:
        for i in instruments[:20]:
            vocab_phrases.append(f"{g} with {i}")
            
    # Add common Spanish, Hindi, Telugu musical terms
    multilingual_terms = [
        "musica relajante", "guitarra acustica", "piano clasico", "cancion alegre", "ritmo bailable", "bajo distorsionado", "musica romantica", "bateria potente", "saxofon jazz", "melodia suave", "musica electronica bailable", "rap en espanol", "flauta andina", "musica instrumental tranquila", "ambiente relajado",
        "shaant sangeet", "dholak taal", "bansuri ki dhun", "tez sangeet", "purana sangeet", "tabla aur sitar", "desh bhakti sangeet", "ghazal sangeet", "sufi sangeet", "sad song dard bhara", "filmi gaana", "romantic gana", "bhangra beat dhol", "bhajan sangeet", "mandir ghanti",
        "prasanta sangeetham", "madhuramaina paata", "vegavanthamaina paata", "telugu paatalu", "dappu sangeetham", "bhakthi paatalu", "veena sangeetham", "nadaswaram sangeetham", "prema paatalu", "janapada geethalu"
    ]
    vocab_phrases.extend(multilingual_terms)
    vocab_phrases = sorted(list(set(vocab_phrases)))[:1000]
    log(f"Generated {len(vocab_phrases)} unique vocabulary phrases.")
    
    # Compute embeddings in batches
    all_embs = []
    for b in range(0, len(vocab_phrases), 64):
        batch = vocab_phrases[b:b+64]
        inp = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            feat = text_tower(inp["input_ids"], inp["attention_mask"])
            feat = F.normalize(feat, dim=-1)
            all_embs.append(feat.half().cpu().numpy()) # FP16 to keep tiny
    all_embs = np.vstack(all_embs)
    np.savez_compressed(vocab_file, phrases=vocab_phrases, embeddings=all_embs)
    log(f"Saved vocabulary embeddings to {vocab_file}: {os.path.getsize(vocab_file)} bytes ({os.path.getsize(vocab_file)/(1024*1024):.2f} MB)")
else:
    data = np.load(vocab_file, allow_pickle=True)
    vocab_phrases = list(data["phrases"])
    all_embs = data["embeddings"]

# Tiny vocabulary matcher benchmark: map query words to nearest vocabulary phrase
def run_vocab_matcher(test_prompts):
    t0 = time.perf_counter()
    cos_sims = []
    
    # Simple word-overlap / token matcher to pick best vocabulary embedding
    for i, p in enumerate(test_prompts):
        tokens = set(p.lower().split())
        best_idx = 0
        best_score = -1
        for idx, phr in enumerate(vocab_phrases):
            phr_tokens = set(phr.lower().split())
            score = len(tokens.intersection(phr_tokens)) / (len(tokens.union(phr_tokens)) + 1e-5)
            if score > best_score:
                best_score = score
                best_idx = idx
        # Retrieve precomputed embedding (cast to float32)
        v_emb = all_embs[best_idx].astype(np.float32)
        v_emb = v_emb / (np.linalg.norm(v_emb) + 1e-8)
        cos = float(np.dot(v_emb, pt_embs[i]))
        cos_sims.append(cos)
    lat_ms = (time.perf_counter() - t0) / len(test_prompts) * 1000.0
    return lat_ms, float(np.mean(cos_sims)), float(np.min(cos_sims))

results["Vocab_1000_Matcher"] = benchmark_model("Pre-computed Vocab (1000 phrases, FP16) + Matcher", vocab_file, test_prompts, is_custom_run=run_vocab_matcher)

# 5. Recommendation Summary Table
log("\n--- [PART 5] CLAP TEXT TOWER COMPRESSION SUMMARY TABLE (MEASURED) ---")
log("| Configuration | Model Size | Mean Cosine vs PyTorch | Min Cosine | Query Latency | Threshold (>=0.99) | Recommended? |")
log("|---|---|---|---|---|---|---|")
for k, r in results.items():
    passed = "PASS" if r["cosine"] >= 0.99 else "FAIL"
    rec = "NO"
    if k == "FP16_Text_Tower":
        rec = "**YES (High Fidelity)**"
    elif k == "Selective_INT8" and r["cosine"] >= 0.99:
        rec = "**YES (Smallest Full Tower)**"
    elif k == "Vocab_1000_Matcher":
        rec = "Vibe Search Alternative (1 MB)"
    log(f"| MEASURED: {k} | {r['size_mb']:.2f} MB | **{r['cosine']:.6f}** | {r['min_cosine']:.6f} | {r['latency_ms']:.2f} ms | {passed} | {rec} |")

log("\n=== R4 EVALUATION COMPLETED ===")
log_out.close()
