import os
import numpy as np
import soundfile as sf

def synthesize_track(track_type, duration_sec=15.0, sr=44100):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = np.zeros(len(t))
    
    if track_type == "ambient_drone":
        # 60 BPM, C Minor (C=65.4Hz, Eb=77.8Hz, G=98.0Hz)
        drone = np.sin(2 * np.pi * 65.41 * t) * 0.4
        drone += np.sin(2 * np.pi * 77.78 * t) * 0.25
        drone += np.sin(2 * np.pi * 98.00 * t) * 0.2
        # filtered noise (rain / atmosphere)
        noise = np.random.randn(len(t)) * 0.05
        # smooth modulation
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
        audio = (drone + noise) * lfo
        
    elif track_type == "fingerstyle_guitar":
        # 75 BPM (beat = 0.8s), A Minor (A2=110, C3=130.8, E3=164.8, A3=220)
        beat_len = int(0.8 * sr)
        freqs = [110.0, 130.81, 164.81, 220.0, 196.0, 164.81, 130.81, 110.0]
        for i, f in enumerate(freqs * 3):
            start = i * (beat_len // 2)
            if start + sr > len(t):
                break
            env = np.exp(-np.linspace(0, 4, sr))
            audio[start:start+sr] += np.sin(2 * np.pi * f * np.linspace(0, 1, sr)) * env * 0.3
            
    elif track_type == "lofi_hiphop":
        # 85 BPM (beat = 0.706s), D Minor Rhodes chords + kick/snare
        beat_s = 60.0 / 85.0
        beat_len = int(beat_s * sr)
        # kick on 1, 3; snare on 2, 4
        num_beats = int(duration_sec / beat_s)
        for b in range(num_beats):
            st = int(b * beat_s * sr)
            if b % 2 == 0:
                # kick (pitch drop 120 -> 40 Hz)
                k_len = int(0.3 * sr)
                kt = np.linspace(0, 0.3, k_len)
                kf = np.linspace(120, 40, k_len)
                audio[st:st+k_len] += np.sin(2 * np.pi * kf * kt) * np.exp(-kt*15) * 0.6
            else:
                # snare (noise burst)
                sn_len = int(0.2 * sr)
                audio[st:st+sn_len] += np.random.randn(sn_len) * np.exp(-np.linspace(0, 5, sn_len)) * 0.4
        # Rhodes chords (Dmin: 146.8, 174.6, 220.0)
        chord = (np.sin(2 * np.pi * 146.83 * t) + np.sin(2 * np.pi * 174.61 * t) + np.sin(2 * np.pi * 220.0 * t)) * 0.15
        audio += chord
        
    elif track_type == "synthwave_cyberpunk":
        # 120 BPM (beat = 0.5s), F# Minor (F#2=92.5Hz), 80s bassline
        beat_s = 60.0 / 120.0
        num_beats = int(duration_sec / beat_s)
        for b in range(num_beats):
            st = int(b * beat_s * sr)
            k_len = int(0.25 * sr)
            audio[st:st+k_len] += np.sin(2 * np.pi * 60 * np.linspace(0, 0.25, k_len)) * 0.5
            # 16th note bass
            for step in range(4):
                bst = st + int(step * 0.25 * beat_s * sr)
                blen = int(0.1 * sr)
                audio[bst:bst+blen] += np.sign(np.sin(2 * np.pi * 92.5 * np.linspace(0, 0.1, blen))) * 0.2
                
    elif track_type == "heavy_trap":
        # 140 BPM, G# Minor (G#1=51.9Hz 808 sub)
        beat_s = 60.0 / 140.0
        num_beats = int(duration_sec / beat_s)
        for b in range(num_beats):
            st = int(b * beat_s * sr)
            if b % 4 == 0:
                # 808 sub boom
                sub_len = int(beat_s * 2 * sr)
                s_t = np.linspace(0, beat_s * 2, min(sub_len, len(t) - st))
                audio[st:st+len(s_t)] += np.sin(2 * np.pi * 51.91 * s_t) * np.exp(-s_t * 1.5) * 0.7
            if b % 2 == 1:
                # hard clap
                cl_len = int(0.15 * sr)
                audio[st:st+cl_len] += np.random.randn(cl_len) * np.exp(-np.linspace(0, 10, cl_len)) * 0.5
                
    elif track_type == "classical_piano":
        # 68 BPM, C Major (C4=261.6, E4=329.6, G4=392.0)
        beat_s = 60.0 / 68.0
        notes = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63]
        for i, f in enumerate(notes * 4):
            st = int(i * 0.5 * beat_s * sr)
            if st + sr > len(t): break
            p_len = min(sr, len(t) - st)
            pt = np.linspace(0, p_len/sr, p_len)
            audio[st:st+p_len] += (np.sin(2*np.pi*f*pt) + 0.3*np.sin(4*np.pi*f*pt)) * np.exp(-pt * 2.5) * 0.35
            
    elif track_type == "funk_disco":
        # 124 BPM, E Minor (E2=82.4Hz slap bass)
        beat_s = 60.0 / 124.0
        num_beats = int(duration_sec / beat_s)
        for b in range(num_beats):
            st = int(b * beat_s * sr)
            k_len = int(0.2 * sr)
            audio[st:st+k_len] += np.sin(2*np.pi*55*np.linspace(0, 0.2, k_len)) * 0.5
            # hihat
            hh_len = int(0.08 * sr)
            audio[st:st+hh_len] += np.random.randn(hh_len) * 0.15
            
    elif track_type == "jazz_swing":
        # 132 BPM, Bb Major (Bb1=58.27, D2=73.4, F2=87.3, G2=98.0 walking bass)
        beat_s = 60.0 / 132.0
        num_beats = int(duration_sec / beat_s)
        walk_f = [58.27, 73.42, 87.31, 98.00]
        for b in range(num_beats):
            st = int(b * beat_s * sr)
            f = walk_f[b % len(walk_f)]
            w_len = int(beat_s * sr)
            wt = np.linspace(0, beat_s, min(w_len, len(t) - st))
            audio[st:st+len(wt)] += np.sin(2 * np.pi * f * wt) * np.exp(-wt * 2.0) * 0.4
            
    elif track_type == "cinematic_strings":
        # 70 BPM, D Major (D3=146.8, F#3=185.0, A3=220.0, D4=293.7)
        strings = np.sin(2 * np.pi * 146.83 * t) * 0.25
        strings += np.sin(2 * np.pi * 184.99 * t) * 0.2
        strings += np.sin(2 * np.pi * 220.00 * t) * 0.2
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * (70.0 / 60.0 / 4.0) * t)
        audio = strings * swell
        
    elif track_type == "reggae_dub":
        # 78 BPM, G Minor (G1=49.0Hz deep sub, offbeat chord)
        beat_s = 60.0 / 78.0
        num_beats = int(duration_sec / beat_s)
        for b in range(num_beats):
            st = int(b * beat_s * sr)
            # bass on 1
            if b % 2 == 0:
                b_len = int(0.5 * sr)
                bt = np.linspace(0, 0.5, b_len)
                audio[st:st+b_len] += np.sin(2 * np.pi * 48.99 * bt) * 0.6
            # offbeat chop on the '&' of the beat
            off_st = st + int(0.5 * beat_s * sr)
            if off_st + int(0.15 * sr) < len(t):
                chop_len = int(0.15 * sr)
                ct = np.linspace(0, 0.15, chop_len)
                audio[off_st:off_st+chop_len] += (np.sin(2*np.pi*196*ct) + np.sin(2*np.pi*233*ct)) * np.exp(-ct*15) * 0.3
                
    # Normalize to -1 dB
    max_val = np.max(np.abs(audio))
    if max_val > 1e-4:
        audio = audio / max_val * 0.89
    # Duplicate to stereo
    stereo = np.stack([audio, audio], axis=1)
    return stereo

def main():
    out_dir = "ai-spike/data/tracks"
    os.makedirs(out_dir, exist_ok=True)
    
    track_meta = [
        ("track_01_ambient_drone.wav", "ambient_drone", 60.0, "C Minor", "Ambient Drone", "rainy late-night drive, meditation, ambient drone, calm synthesizer"),
        ("track_02_fingerstyle_guitar.wav", "fingerstyle_guitar", 75.0, "A Minor", "Folk / Acoustic", "acoustic guitar, melancholic folk, sad arpeggio, solo strings"),
        ("track_03_lofi_hiphop.wav", "lofi_hiphop", 85.0, "D Minor", "Lo-Fi Hip Hop", "lofi hip hop beat, chill study beats, rhodes piano, boom bap kick"),
        ("track_04_synthwave_cyberpunk.wav", "synthwave_cyberpunk", 120.0, "F# Minor", "Synthwave", "synthwave, fast driving electronic, neon cyberpunk, 80s bassline"),
        ("track_05_heavy_trap.wav", "heavy_trap", 140.0, "G# Minor", "Trap", "heavy 808 bass, fast trap beat, hihat rolls, aggressive workout"),
        ("track_06_classical_piano.wav", "classical_piano", 68.0, "C Major", "Classical", "classical piano solo, peaceful morning, gentle acoustic nocturne"),
        ("track_07_funk_disco.wav", "funk_disco", 124.0, "E Minor", "Funk / Disco", "upbeat funk groove, dance disco, slap bass, cheerful energetic"),
        ("track_08_jazz_swing.wav", "jazz_swing", 132.0, "Bb Major", "Jazz", "jazz swing, walking bass, acoustic cocktail lounge, ride cymbal"),
        ("track_09_cinematic_strings.wav", "cinematic_strings", 70.0, "D Major", "Cinematic", "cinematic orchestral strings, dramatic film score, cello violin swell"),
        ("track_10_reggae_dub.wav", "reggae_dub", 78.0, "G Minor", "Reggae / Dub", "reggae dub, summer island vibe, deep sub bass, offbeat skank"),
    ]
    
    print(f"Generating {len(track_meta)} reference audio tracks...")
    manifest_lines = ["Filename,TrackType,GroundTruthBPM,GroundTruthKey,Genre,MoodTags"]
    for fname, t_type, bpm, key, genre, tags in track_meta:
        path = os.path.join(out_dir, fname)
        audio = synthesize_track(t_type, duration_sec=15.0)
        sf.write(path, audio, 44100)
        manifest_lines.append(f"{fname},{t_type},{bpm},{key},{genre},\"{tags}\"")
        print(f"  Wrote {path} ({bpm} BPM, {key})")
        
    with open("ai-spike/data/reference_manifest.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")
    print("Reference tracks generated successfully!")

if __name__ == "__main__":
    main()
