//! Audio file metadata extraction using Lofty, Blake3, and Image.

use std::ffi::{c_char, CStr, CString};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use serde::Serialize;

use lofty::file::{AudioFile, FileType, TaggedFileExt};
use lofty::picture::{Picture, PictureType};
use lofty::probe::Probe;
use lofty::tag::{Accessor, ItemKey, Tag};

#[derive(Serialize)]
pub struct ArtworkMeta {
    pub hash: String,
    pub path: String,
    pub width: i32,
    pub height: i32,
}

#[derive(Serialize)]
pub struct MetaSuccess {
    pub ok: bool,
    pub title: Option<String>,
    pub artists: Vec<String>,
    pub album_artist: Option<String>,
    pub album: Option<String>,
    pub genres: Vec<String>,
    pub track_no: Option<i32>,
    pub disc_no: Option<i32>,
    pub year: Option<i32>,
    pub duration_ms: i64,
    pub bitrate_kbps: Option<i32>,
    pub sample_rate: Option<i32>,
    pub channels: Option<i32>,
    pub bit_depth: Option<i32>,
    pub codec: String,
    pub lossless: bool,
    pub rg_track_gain_db: Option<f64>,
    pub rg_track_peak: Option<f64>,
    pub rg_album_gain_db: Option<f64>,
    pub rg_album_peak: Option<f64>,
    pub has_embedded_lyrics: bool,
    pub artwork: Option<ArtworkMeta>,
    pub fingerprint: Option<String>,
    pub meta_version: i32,
}

#[derive(Serialize)]
pub struct MetaFailure {
    pub ok: bool,
    pub error: String,
}

/// Compute 64 KiB front + 64 KiB back blake3 fingerprint.
pub fn compute_fingerprint(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let size = metadata.len();
    let chunk_size = 64 * 1024; // 64 KiB
    let threshold = 128 * 1024; // 128 KiB

    let mut hasher = blake3::Hasher::new();
    if size <= threshold {
        let mut buf = Vec::with_capacity(size as usize);
        file.read_to_end(&mut buf)?;
        hasher.update(&buf);
    } else {
        let mut first_chunk = vec![0u8; chunk_size as usize];
        file.read_exact(&mut first_chunk)?;
        hasher.update(&first_chunk);

        file.seek(SeekFrom::End(-(chunk_size as i64)))?;
        let mut last_chunk = vec![0u8; chunk_size as usize];
        file.read_exact(&mut last_chunk)?;
        hasher.update(&last_chunk);
    }
    let hash = hasher.finalize().to_hex().to_string();
    Ok(format!("{}:{}", size, hash))
}

fn parse_gain_db(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    let without_unit = if let Some(stripped) = trimmed
        .strip_suffix("dB")
        .or_else(|| trimmed.strip_suffix("db"))
        .or_else(|| trimmed.strip_suffix("DB"))
    {
        stripped.trim()
    } else {
        trimmed
    };
    without_unit.parse::<f64>().ok()
}

fn parse_peak(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

fn parse_year(s: &str) -> Option<i32> {
    let s = s.trim();
    if s.len() >= 4 {
        if let Ok(y) = s[0..4].parse::<i32>() {
            if (1000..=2999).contains(&y) {
                return Some(y);
            }
        }
    }
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len().saturating_sub(3) {
        if chars[i..i + 4].iter().all(|c| c.is_ascii_digit()) {
            let slice: String = chars[i..i + 4].iter().collect();
            if let Ok(y) = slice.parse::<i32>() {
                if (1000..=2999).contains(&y) {
                    return Some(y);
                }
            }
        }
    }
    None
}

fn extract_multi_strings(tag: &Tag, key: ItemKey) -> Vec<String> {
    let mut results = Vec::new();
    for item in tag.items() {
        if item.key() == key {
            if let Some(text) = item.value().text() {
                // If text contains null characters (\0), split on null
                for piece in text.split('\0') {
                    let t = piece.trim();
                    if !t.is_empty() && !results.iter().any(|existing: &String| existing == t) {
                        results.push(t.to_string());
                    }
                }
            }
        }
    }
    results
}

fn detect_codec(file_type: FileType, path: &Path) -> String {
    match file_type {
        FileType::Mpeg => "mp3".to_string(),
        FileType::Flac => "flac".to_string(),
        FileType::Aac => "aac".to_string(),
        FileType::Vorbis => "vorbis".to_string(),
        FileType::Opus => "opus".to_string(),
        FileType::Wav => "wav".to_string(),
        FileType::Aiff => "aiff".to_string(),
        FileType::Ape => "ape".to_string(),
        FileType::WavPack => "wavpack".to_string(),
        FileType::Mp4 => {
            if let Ok(mut f) = File::open(path) {
                if let Ok(mp4_file) = lofty::mp4::Mp4File::read_from(&mut f, lofty::config::ParseOptions::default()) {
                    if mp4_file.properties().codec() == Some(lofty::mp4::Mp4Codec::ALAC) {
                        return "alac".to_string();
                    }
                }
            }
            "aac".to_string()
        }
        _ => "other".to_string(),
    }
}

fn process_artwork(pictures: &[Picture], art_dir: Option<&Path>) -> Option<ArtworkMeta> {
    let art_dir = art_dir?;
    if pictures.is_empty() {
        return None;
    }

    // Prefer front cover, otherwise first picture
    let pic = pictures
        .iter()
        .find(|p| p.pic_type() == PictureType::CoverFront)
        .or_else(|| pictures.first())?;

    let bytes = pic.data();
    if bytes.is_empty() {
        return None;
    }

    let hash = blake3::hash(bytes).to_hex().to_string();
    if let Err(e) = fs::create_dir_all(art_dir) {
        log::error!("[unbound-audio] Failed to create artwork dir {:?}: {e}", art_dir);
        return None;
    }

    let jpg_path = art_dir.join(format!("{}.jpg", hash));
    if jpg_path.exists() {
        if let Ok((w, h)) = image::image_dimensions(&jpg_path) {
            return Some(ArtworkMeta {
                hash,
                path: jpg_path.to_string_lossy().into_owned(),
                width: w as i32,
                height: h as i32,
            });
        }
    }

    // Attempt decode
    match image::load_from_memory(bytes) {
        Ok(img) => {
            let (w, h) = (img.width(), img.height());
            let longest = w.max(h);
            let img_to_save = if longest > 600 {
                img.resize(600, 600, image::imageops::FilterType::Lanczos3)
            } else {
                img
            };
            let rgb = img_to_save.to_rgb8();
            let (final_w, final_h) = (rgb.width(), rgb.height());

            let write_res = (|| -> Result<(), Box<dyn std::error::Error>> {
                let file = File::create(&jpg_path)?;
                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 85);
                encoder.encode(&rgb, final_w, final_h, image::ExtendedColorType::Rgb8)?;
                Ok(())
            })();

            match write_res {
                Ok(()) => Some(ArtworkMeta {
                    hash,
                    path: jpg_path.to_string_lossy().into_owned(),
                    width: final_w as i32,
                    height: final_h as i32,
                }),
                Err(e) => {
                    log::error!("[unbound-audio] Failed to encode artwork to {:?}: {e}", jpg_path);
                    None
                }
            }
        }
        Err(e) => {
            log::warn!("[unbound-audio] Image decode failed ({e}); saving raw bytes");
            let ext = match pic.mime_type() {
                Some(lofty::picture::MimeType::Png) => "png",
                Some(lofty::picture::MimeType::Jpeg) => "jpg",
                Some(lofty::picture::MimeType::Bmp) => "bmp",
                Some(lofty::picture::MimeType::Gif) => "gif",
                Some(lofty::picture::MimeType::Tiff) => "tiff",
                _ => "bin",
            };
            let raw_path = art_dir.join(format!("{}.{}", hash, ext));
            if !raw_path.exists() {
                let _ = fs::write(&raw_path, bytes);
            }
            Some(ArtworkMeta {
                hash,
                path: raw_path.to_string_lossy().into_owned(),
                width: 0,
                height: 0,
            })
        }
    }
}

pub fn read_metadata(path: &Path, art_dir: Option<&Path>) -> Result<MetaSuccess, String> {
    let fingerprint = compute_fingerprint(path).map_err(|e| format!("Fingerprint error: {e}"))?;

    let tagged_file = Probe::open(path)
        .map_err(|e| format!("Probe open error: {e}"))?
        .read()
        .map_err(|e| format!("Probe read error: {e}"))?;

    let properties = tagged_file.properties();
    let file_type = tagged_file.file_type();
    let codec = detect_codec(file_type, path);
    let lossless = matches!(
        codec.as_str(),
        "flac" | "wav" | "aiff" | "ape" | "wavpack" | "alac"
    );

    // Audio properties
    let duration_ms = properties.duration().as_millis() as i64;
    let bitrate_kbps = properties
        .audio_bitrate()
        .or_else(|| properties.overall_bitrate())
        .map(|b| b as i32);
    let sample_rate = properties.sample_rate().map(|r| r as i32);
    let channels = properties.channels().map(|c| c as i32);
    let bit_depth = properties.bit_depth().map(|d| d as i32);

    let tag = tagged_file.primary_tag().or_else(|| tagged_file.first_tag());

    let (
        title,
        artists,
        album_artist,
        album,
        genres,
        track_no,
        disc_no,
        year,
        rg_track_gain_db,
        rg_track_peak,
        rg_album_gain_db,
        rg_album_peak,
        has_embedded_lyrics,
        artwork,
    ) = if let Some(t) = tag {
        let title = t.title().map(|s| s.to_string());
        let artists = extract_multi_strings(t, ItemKey::TrackArtist);
        let album_artist = t.get_string(ItemKey::AlbumArtist).map(|s| s.to_string());
        let album = t.album().map(|s| s.to_string());
        let genres = extract_multi_strings(t, ItemKey::Genre);
        let track_no = t.track().map(|n| n as i32);
        let disc_no = t.disk().map(|n| n as i32);

        let year = t
            .date()
            .map(|d| d.year as i32)
            .or_else(|| t.get_string(ItemKey::Year).and_then(parse_year))
            .or_else(|| t.get_string(ItemKey::RecordingDate).and_then(parse_year))
            .or_else(|| t.get_string(ItemKey::OriginalReleaseDate).and_then(parse_year));

        let rg_track_gain = t
            .get_string(ItemKey::ReplayGainTrackGain)
            .and_then(parse_gain_db);
        let rg_track_peak = t
            .get_string(ItemKey::ReplayGainTrackPeak)
            .and_then(parse_peak);
        let rg_album_gain = t
            .get_string(ItemKey::ReplayGainAlbumGain)
            .and_then(parse_gain_db);
        let rg_album_peak = t
            .get_string(ItemKey::ReplayGainAlbumPeak)
            .and_then(parse_peak);

        let has_lyrics = t.get_string(ItemKey::Lyrics).is_some()
            || t.get_string(ItemKey::UnsyncLyrics).is_some();
        let artwork = process_artwork(t.pictures(), art_dir);

        (
            title,
            artists,
            album_artist,
            album,
            genres,
            track_no,
            disc_no,
            year,
            rg_track_gain,
            rg_track_peak,
            rg_album_gain,
            rg_album_peak,
            has_lyrics,
            artwork,
        )
    } else {
        (
            None,
            Vec::new(),
            None,
            None,
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
        )
    };

    Ok(MetaSuccess {
        ok: true,
        title,
        artists,
        album_artist,
        album,
        genres,
        track_no,
        disc_no,
        year,
        duration_ms,
        bitrate_kbps,
        sample_rate,
        channels,
        bit_depth,
        codec,
        lossless,
        rg_track_gain_db,
        rg_track_peak,
        rg_album_gain_db,
        rg_album_peak,
        has_embedded_lyrics,
        artwork,
        fingerprint: Some(fingerprint),
        meta_version: 1,
    })
}

// ---------------------------------------------------------------------
// C ABI exports
// ---------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn meta_read_json(path: *const c_char, art_dir: *const c_char) -> *mut c_char {
    let result = std::panic::catch_unwind(|| {
        if path.is_null() {
            let fail = MetaFailure {
                ok: false,
                error: "Null path pointer".into(),
            };
            return serde_json::to_string(&fail).unwrap_or_else(|_| "{\"ok\":false}".into());
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                let fail = MetaFailure {
                    ok: false,
                    error: format!("Non-UTF8 path: {e}"),
                };
                return serde_json::to_string(&fail).unwrap_or_else(|_| "{\"ok\":false}".into());
            }
        };

        let art_dir_buf = if !art_dir.is_null() {
            match unsafe { CStr::from_ptr(art_dir) }.to_str() {
                Ok(s) if !s.is_empty() => Some(PathBuf::from(s)),
                _ => None,
            }
        } else {
            None
        };

        match read_metadata(Path::new(path_str), art_dir_buf.as_deref()) {
            Ok(success) => serde_json::to_string(&success)
                .unwrap_or_else(|e| format!("{{\"ok\":false,\"error\":\"JSON serialize error: {e}\"}}")),
            Err(e) => {
                let fail = MetaFailure {
                    ok: false,
                    error: e,
                };
                serde_json::to_string(&fail).unwrap_or_else(|_| "{\"ok\":false}".into())
            }
        }
    });

    let json = result.unwrap_or_else(|_| {
        log::error!("[unbound-audio] Panic in meta_read_json");
        "{\"ok\":false,\"error\":\"Internal panic while reading metadata\"}".to_string()
    });

    CString::new(json)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn meta_free_string(ptr: *mut c_char) {
    let _ = std::panic::catch_unwind(|| {
        if !ptr.is_null() {
            unsafe {
                drop(CString::from_raw(ptr));
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use lofty::config::WriteOptions;
    use lofty::tag::{ItemKey, ItemValue, Tag, TagExt, TagItem, TagType};

    fn create_dummy_wav(path: &Path) {
        let rate = 44_100u32;
        let samples = rate / 4; // 250ms
        let mut pcm: Vec<u8> = Vec::new();
        for _ in 0..samples {
            let v = 1000i16;
            pcm.extend_from_slice(&v.to_le_bytes());
            pcm.extend_from_slice(&v.to_le_bytes());
        }
        let mut w: Vec<u8> = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&2u16.to_le_bytes());
        w.extend_from_slice(&rate.to_le_bytes());
        w.extend_from_slice(&(rate * 4).to_le_bytes());
        w.extend_from_slice(&4u16.to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        w.extend_from_slice(&pcm);
        fs::write(path, w).unwrap();
    }

    #[test]
    fn test_tagged_wav_multi_artist_and_replaygain() {
        let temp_dir = std::env::temp_dir().join("unbound_test_meta_1");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let wav_path = temp_dir.join("test_tagged.wav");
        create_dummy_wav(&wav_path);

        let mut tag = Tag::new(TagType::Id3v2);
        tag.set_title("Starlight".to_string());
        tag.push(TagItem::new(ItemKey::TrackArtist, ItemValue::Text("Muse".into())));
        tag.push(TagItem::new(ItemKey::TrackArtist, ItemValue::Text("Matt Bellamy".into())));
        tag.push(TagItem::new(ItemKey::AlbumTitle, ItemValue::Text("Black Holes".into())));
        tag.push(TagItem::new(ItemKey::AlbumArtist, ItemValue::Text("Muse".into())));
        tag.push(TagItem::new(ItemKey::Genre, ItemValue::Text("Rock".into())));
        tag.push(TagItem::new(ItemKey::Genre, ItemValue::Text("Alternative".into())));
        tag.push(TagItem::new(ItemKey::TrackNumber, ItemValue::Text("2".into())));
        tag.push(TagItem::new(ItemKey::DiscNumber, ItemValue::Text("1".into())));
        tag.push(TagItem::new(ItemKey::RecordingDate, ItemValue::Text("2006".into())));
        tag.push(TagItem::new(ItemKey::ReplayGainTrackGain, ItemValue::Text("-6.50 dB".into())));
        tag.push(TagItem::new(ItemKey::ReplayGainTrackPeak, ItemValue::Text("0.9882".into())));
        tag.push(TagItem::new(ItemKey::ReplayGainAlbumGain, ItemValue::Text("-5.20 dB".into())));
        tag.push(TagItem::new(ItemKey::ReplayGainAlbumPeak, ItemValue::Text("1.0000".into())));
        tag.push(TagItem::new(ItemKey::UnsyncLyrics, ItemValue::Text("Hold me in your arms".into())));
        tag.save_to_path(&wav_path, WriteOptions::default()).unwrap();

        let meta = read_metadata(&wav_path, None).unwrap();
        assert_eq!(meta.title.as_deref(), Some("Starlight"));
        assert_eq!(meta.artists, vec!["Muse", "Matt Bellamy"]);
        assert_eq!(meta.album.as_deref(), Some("Black Holes"));
        assert_eq!(meta.album_artist.as_deref(), Some("Muse"));
        assert_eq!(meta.genres, vec!["Rock", "Alternative"]);
        assert_eq!(meta.track_no, Some(2));
        assert_eq!(meta.disc_no, Some(1));
        assert_eq!(meta.year, Some(2006));
        assert_eq!(meta.rg_track_gain_db, Some(-6.50));
        assert_eq!(meta.rg_track_peak, Some(0.9882));
        assert_eq!(meta.rg_album_gain_db, Some(-5.20));
        assert_eq!(meta.rg_album_peak, Some(1.0000));
        assert!(meta.has_embedded_lyrics);
        assert_eq!(meta.codec, "wav");
        assert!(meta.lossless);
        assert_eq!(meta.meta_version, 1);
    }

    #[test]
    fn test_artist_no_split_on_special_characters() {
        let temp_dir = std::env::temp_dir().join("unbound_test_meta_2");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let wav_path = temp_dir.join("acdc.wav");
        create_dummy_wav(&wav_path);

        let mut tag = Tag::new(TagType::Id3v2);
        tag.push(TagItem::new(ItemKey::TrackArtist, ItemValue::Text("AC/DC".into())));
        tag.push(TagItem::new(ItemKey::TrackArtist, ItemValue::Text("Simon & Garfunkel".into())));
        tag.save_to_path(&wav_path, WriteOptions::default()).unwrap();

        let meta = read_metadata(&wav_path, None).unwrap();
        assert_eq!(meta.artists, vec!["AC/DC", "Simon & Garfunkel"]);
    }

    #[test]
    fn test_embedded_art_dedupe_and_downscale() {
        let temp_dir = std::env::temp_dir().join("unbound_test_meta_art");
        let _ = fs::remove_dir_all(&temp_dir);
        let art_dir = temp_dir.join("art_cache");
        fs::create_dir_all(&temp_dir).unwrap();

        // Generate 700x700 image
        let mut img = image::RgbaImage::new(700, 700);
        for p in img.pixels_mut() {
            *p = image::Rgba([200, 50, 100, 255]);
        }
        let mut png_bytes = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png).unwrap();

        let wav1 = temp_dir.join("track1.wav");
        let wav2 = temp_dir.join("track2.wav");
        create_dummy_wav(&wav1);
        create_dummy_wav(&wav2);

        let mut tag1 = Tag::new(TagType::Id3v2);
        let mut pic1 = Picture::from_reader(&mut std::io::Cursor::new(&png_bytes)).unwrap();
        pic1.set_pic_type(PictureType::CoverFront);
        tag1.push_picture(pic1);
        tag1.save_to_path(&wav1, WriteOptions::default()).unwrap();

        let mut tag2 = Tag::new(TagType::Id3v2);
        let mut pic2 = Picture::from_reader(&mut std::io::Cursor::new(&png_bytes)).unwrap();
        pic2.set_pic_type(PictureType::CoverFront);
        tag2.push_picture(pic2);
        tag2.save_to_path(&wav2, WriteOptions::default()).unwrap();

        let meta1 = read_metadata(&wav1, Some(&art_dir)).unwrap();
        let meta2 = read_metadata(&wav2, Some(&art_dir)).unwrap();

        let art1 = meta1.artwork.expect("art1 should exist");
        let art2 = meta2.artwork.expect("art2 should exist");

        // Both must share identical hash and path
        assert_eq!(art1.hash, art2.hash);
        assert_eq!(art1.path, art2.path);
        // Longest side <= 600
        assert_eq!(art1.width, 600);
        assert_eq!(art1.height, 600);
        assert!(Path::new(&art1.path).exists());
    }

    #[test]
    fn test_corrupt_and_non_audio_files_return_ok_false() {
        let temp_dir = std::env::temp_dir().join("unbound_test_meta_corrupt");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let zero_file = temp_dir.join("zero.mp3");
        fs::write(&zero_file, b"").unwrap();

        let corrupt_file = temp_dir.join("corrupt.flac");
        fs::write(&corrupt_file, b"NOT AN AUDIO FILE AT ALL").unwrap();

        let missing_file = temp_dir.join("nonexistent.wav");

        let c_zero = CString::new(zero_file.to_str().unwrap()).unwrap();
        let ptr_zero = meta_read_json(c_zero.as_ptr(), std::ptr::null());
        let json_zero = unsafe { CStr::from_ptr(ptr_zero) }.to_str().unwrap();
        assert!(json_zero.contains("\"ok\":false"));
        meta_free_string(ptr_zero);

        let c_corrupt = CString::new(corrupt_file.to_str().unwrap()).unwrap();
        let ptr_corrupt = meta_read_json(c_corrupt.as_ptr(), std::ptr::null());
        let json_corrupt = unsafe { CStr::from_ptr(ptr_corrupt) }.to_str().unwrap();
        assert!(json_corrupt.contains("\"ok\":false"));
        meta_free_string(ptr_corrupt);

        let c_missing = CString::new(missing_file.to_str().unwrap()).unwrap();
        let ptr_missing = meta_read_json(c_missing.as_ptr(), std::ptr::null());
        let json_missing = unsafe { CStr::from_ptr(ptr_missing) }.to_str().unwrap();
        assert!(json_missing.contains("\"ok\":false"));
        meta_free_string(ptr_missing);
    }

    #[test]
    fn test_fingerprint_stability_and_sensitivity() {
        let temp_dir = std::env::temp_dir().join("unbound_test_meta_fp");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let file_path = temp_dir.join("sample.bin");
        // 200 KiB file
        let mut data = vec![42u8; 200 * 1024];
        fs::write(&file_path, &data).unwrap();

        let fp1 = compute_fingerprint(&file_path).unwrap();
        let fp2 = compute_fingerprint(&file_path).unwrap();
        assert_eq!(fp1, fp2);
        assert!(fp1.starts_with("204800:"));

        // Modifying byte in middle (offset 100 KiB) should NOT change fingerprint (only first/last 64 KiB hashed)
        data[100 * 1024] = 99;
        fs::write(&file_path, &data).unwrap();
        let fp_mid = compute_fingerprint(&file_path).unwrap();
        assert_eq!(fp1, fp_mid);

        // Modifying byte in first 64 KiB (offset 10) MUST change fingerprint
        data[10] = 77;
        fs::write(&file_path, &data).unwrap();
        let fp_first = compute_fingerprint(&file_path).unwrap();
        assert_ne!(fp1, fp_first);

        // Modifying byte in last 64 KiB (offset 190 KiB) MUST change fingerprint
        data[10] = 42; // restore first
        data[190 * 1024] = 88;
        fs::write(&file_path, &data).unwrap();
        let fp_last = compute_fingerprint(&file_path).unwrap();
        assert_ne!(fp1, fp_last);
    }

    #[test]
    fn test_unicode_path_with_spaces() {
        let temp_dir = std::env::temp_dir().join("unbound_test_meta_unicode");
        let _ = fs::remove_dir_all(&temp_dir);
        let sub_dir = temp_dir.join("UNBOUND 🎵 测试 música");
        fs::create_dir_all(&sub_dir).unwrap();

        let wav_path = sub_dir.join("my song.wav");
        create_dummy_wav(&wav_path);

        let mut tag = Tag::new(TagType::Id3v2);
        tag.set_title("Unicode Song".to_string());
        tag.save_to_path(&wav_path, WriteOptions::default()).unwrap();

        let c_path = CString::new(wav_path.to_str().unwrap()).unwrap();
        let ptr = meta_read_json(c_path.as_ptr(), std::ptr::null());
        assert!(!ptr.is_null());
        let json_str = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert!(json_str.contains("\"ok\":true"));
        assert!(json_str.contains("\"title\":\"Unicode Song\""));
        meta_free_string(ptr);
    }
}
