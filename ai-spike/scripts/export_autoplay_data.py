#!/usr/bin/env python3
import os
import json
import csv
import numpy as np

os.makedirs("ai-spike/autoplay-rs/data", exist_ok=True)

# 1. Load real embeddings
cache = np.load("ai-spike/models/clap/real_audio_embeddings_cache.npz")
embs = cache["embs_py"] # [152, 512] float32
valid_indices = cache["valid_indices"].tolist()

# 2. Load manifest
with open("ai-spike/data/real/manifest.csv", "r", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))
manifest = [reader[i] for i in valid_indices]

# 3. Identify clusters
liked_cluster = [] # Jazz, Ragtime, Lo-fi, Classical
skipped_cluster = [] # Metal, Trap, Heavy Rock
other_cluster = []

for idx, tr in enumerate(manifest):
    tags = (tr.get("genre", "") + " " + tr.get("tags", "")).lower()
    if any(k in tags for k in ["jazz", "ragtime", "lo-fi", "classical", "piano"]):
        liked_cluster.append(idx)
    elif any(k in tags for k in ["metal", "trap", "phonk", "grunge"]):
        skipped_cluster.append(idx)
    else:
        other_cluster.append(idx)

print(f"Total embeddings: {len(embs)}")
print(f"Liked cluster tracks: {len(liked_cluster)}")
print(f"Skipped cluster tracks: {len(skipped_cluster)}")
print(f"Other tracks: {len(other_cluster)}")

# Save binary embeddings: [N, 512] f32
embs.astype(np.float32).tofile("ai-spike/autoplay-rs/data/real_embeddings_152.bin")

# Save cluster metadata JSON
meta = {
    "n_tracks": len(embs),
    "dim": 512,
    "liked_indices": liked_cluster,
    "skipped_indices": skipped_cluster,
    "other_indices": other_cluster,
    "track_titles": [tr["title"] for tr in manifest],
    "track_genres": [tr["genre"] for tr in manifest]
}
with open("ai-spike/autoplay-rs/data/clusters.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Saved real_embeddings_152.bin and clusters.json to ai-spike/autoplay-rs/data/")
