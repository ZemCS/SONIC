import os
import json
import re
import traceback
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from pathlib import Path
from datetime import datetime
from mutagen.easyid3 import EasyID3
import essentia.standard as es
import lyricsgenius
import safetensors.torch
from faster_whisper import WhisperModel
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    Wav2Vec2FeatureExtractor,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
)
import logging
from tqdm import tqdm
from langdetect import detect
from deep_translator import GoogleTranslator

# Suppress Transformers logging
transformers.logging.set_verbosity_error()

# Suppress Essentia logs
import essentia
essentia.log.infoActive = False
essentia.log.warningActive = False

from config import config, UI
from database import fused_inferences, audio_inferences, lyrics_inferences

class MoodViT(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3, freeze_backbone=False):
        super(MoodViT, self).__init__()
        
        # Load pre-trained ViT
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Option to freeze backbone
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
                
        # Replace the classifier with your sophisticated head
        feature_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Initialize the new layers (matches the retraining script logic)
        for module in self.vit.heads:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.vit(x)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1)
        )

    def forward(self, last_hidden_state):
        weights = self.attention(last_hidden_state)
        return torch.sum(weights * last_hidden_state, dim=1)

class CustomAudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.audio_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.pooling = AttentionPooling(self.audio_model.config.hidden_size)
        in_dim = self.audio_model.config.hidden_size + 2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, audio_values, va_tensor):
        outputs = self.audio_model(input_values=audio_values)
        pooled = self.pooling(outputs.last_hidden_state)
        combined_features = torch.cat((pooled, va_tensor), dim=1)
        return self.classifier(combined_features)

class MultimodalMoodClassifier:
    def __init__(self):
        self.device = config.DEVICE
        print(f"Initializing Multimodal Pipeline on: {self.device.type.upper()}")

        GENIUS_CLIENT_ID = os.getenv("GENIUS_CLIENT_ID")
        GENIUS_CLIENT_SECRET = os.getenv("GENIUS_CLIENT_SECRET")

        self.genius = None
        if (
            config.GENIUS_ACCESS_TOKEN
            and config.GENIUS_ACCESS_TOKEN != "YOUR_TOKEN_HERE"
        ):
            self.genius = lyricsgenius.Genius(config.GENIUS_ACCESS_TOKEN)
        elif GENIUS_CLIENT_ID and GENIUS_CLIENT_SECRET:
            try:
                import requests
                print("Authenticating with Genius via Client ID and Secret...")
                res = requests.post(
                    "https://api.genius.com/oauth/token",
                    data={
                        "client_id": GENIUS_CLIENT_ID,
                        "client_secret": GENIUS_CLIENT_SECRET,
                        "grant_type": "client_credentials",
                    },
                )
                if res.status_code == 200:
                    token = res.json().get("access_token")
                    self.genius = lyricsgenius.Genius(token)
                else:
                    print(f"Genius auth failed: {res.text}")
            except Exception as e:
                print(f"Exception during Genius auth: {e}")

        if self.genius:
            self.genius.verbose = False
            self.genius.remove_section_headers = False
        else:
            print("Warning: Genius not properly configured. Falling back strictly to Whisper AI.")

        # Correct initialization logic
        self._init_essentia_models()
        self._load_and_verify_labels()
        self._initialize_models()

        # Initialize Faster Whisper
        print(UI.info("Loading Whisper AI Model..."))
        self.whisper_model = WhisperModel("base", device=self.device.type, compute_type="float32")
        print(UI.success("Whisper AI Model Loaded."))

    def _init_essentia_models(self):
        try:
            self.audio_loader = es.MonoLoader(sampleRate=16000)
            self.embedder = es.TensorflowPredictMusiCNN(
                graphFilename=config.EMBEDDING_MODEL, output="model/dense/BiasAdd"
            )
            self.va_predictor = es.TensorflowPredict2D(
                graphFilename=config.VA_MODEL, output="model/Identity"
            )
        except RuntimeError as e:
            raise RuntimeError(f"CRITICAL: Failed to load Essentia models. Ensure .pb files exist. {e}")

    def _load_and_verify_labels(self):
        label_file = os.path.join(config.TEXT_MODEL_PATH, "id_to_label.json")
        with open(label_file, "r") as f:
            raw_labels = json.load(f)
            self.id_to_label = {}
            for k, v in raw_labels.items():
                mood = str(v).lower()
                if mood == "joy":
                    mood = "joyful"
                self.id_to_label[int(k)] = mood
        self.num_classes = len(self.id_to_label)

    def _initialize_models(self):
        print("Loading Text Classification Model...")
        cfg = RobertaConfig.from_pretrained(config.TEXT_MODEL_PATH)
        cfg.num_labels = self.num_classes
        cfg.max_position_embeddings = 514
        cfg.type_vocab_size = 1

        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_PATH)
        self.text_model = (
            RobertaForSequenceClassification.from_pretrained(
                config.TEXT_MODEL_PATH, config=cfg, ignore_mismatched_sizes=True
            )
            .to(self.device)
            .eval()
        )

        print("Loading Audio Feature Extractor and Base Model...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.audio_model = CustomAudioClassifier(num_classes=self.num_classes).to(self.device)

        weight_path_safe = os.path.join(config.AUDIO_MODEL_PATH, "model.safetensors")
        weight_path_bin = os.path.join(config.AUDIO_MODEL_PATH, "pytorch_model.bin")
        if os.path.exists(weight_path_safe):
            state_dict = safetensors.torch.load_file(weight_path_safe)
        else:
            state_dict = torch.load(weight_path_bin, map_location="cpu")

        self.audio_model.load_state_dict(state_dict, strict=False)
        self.audio_model.eval()

    def clean_metadata(self, filename):
        name_no_ext = os.path.splitext(filename)[0].replace("_-_", " - ").replace("_", " ")
        junk_tags = r"(?i)(\(official.*\)|\(lyric.*\)|\(music video.*\)|\(hd.*\)|\(remaster.*\)|\(cover.*\)|\(prod\..*\)|official.*video|hd remaster)"
        clean_name = re.sub(junk_tags, "", name_no_ext).strip(" -[]()")
        artist, title = "", clean_name
        if " - " in clean_name:
            parts = clean_name.split(" - ", 1)
            artist, title = parts[0].strip(), parts[1].strip()
        return artist, title, clean_name

    def fetch_lyrics(self, filepath):
        filename = os.path.basename(filepath)
        artist, title, full_query = self.clean_metadata(filename)
        song = None
        if self.genius:
            try:
                print(f"  > Querying Genius for Title: '{title}', Artist: '{artist}'")
                song = self.genius.search_song(title, artist) if artist else self.genius.search_song(full_query)
            except Exception as e:
                print(f"  > Genius API Error: {e}")

        if song:
            return song.lyrics, "Genius API"

        try:
            segments, _ = self.whisper_model.transcribe(filepath, beam_size=5, vad_filter=True)
            lyrics = " ".join([segment.text for segment in segments]).strip()
            if not lyrics:
                raise ValueError("Empty transcription")
            return lyrics, "Whisper AI"
        except Exception:
            return "[Instrumental/No Lyrics Found]", "None"

    def _get_valence_arousal(self, audio_path):
        try:
            self.audio_loader.configure(filename=audio_path)
            audio = self.audio_loader()
            va_preds = self.va_predictor(self.embedder(audio))
            va_mean = np.mean(va_preds, axis=0)
            return va_mean[0], va_mean[1]
        except Exception:
            return 5.0, 5.0

    def aggregate_confidence_weighted_chunks(self, logits_list, threshold=0.40):
        if not len(logits_list):
            return np.zeros(self.num_classes)
        weighted_logits = np.zeros(self.num_classes)
        total_weight = 0.0
        for l in logits_list:
            exp_l = np.exp(l - np.max(l))
            probs = exp_l / np.sum(exp_l)
            confidence = np.max(probs)
            if confidence < threshold:
                continue
            weight = confidence**2
            weighted_logits += l * weight
            total_weight += weight
        if total_weight == 0.0:
            return np.mean(logits_list, axis=0)
        return weighted_logits / total_weight

    def _normalize_logits(self, logits):
        std = np.std(logits)
        if std == 0:
            return logits - np.mean(logits)
        return (logits - np.mean(logits)) / std

    def analyze_track(self, audio_path):
        filename = os.path.basename(audio_path)
        print(f"\n[Analysing] {filename}")
        print(f"  > Fetching lyrics...")
        lyrics_text, lyrics_source = self.fetch_lyrics(audio_path)
        detected_lang = None
 
        original_lyrics = lyrics_text
        translated_lyrics = None
        if lyrics_text and lyrics_text != "[Instrumental/No Lyrics Found]":
            if not detected_lang:
                try:
                    detected_lang = detect(lyrics_text)
                except Exception:
                    detected_lang = "unknown"
 
            if detected_lang != "en" and detected_lang != "unknown":
                print(f"  > Detected language: {detected_lang}. Translating to English...")
                try:
                    translated_lyrics = GoogleTranslator(source="auto", target="en").translate(lyrics_text)
                    lyrics_text = translated_lyrics
                    
                    # --- NEW DEBUGGING CODE: Output translated lyrics to a .txt file ---
                    # Create a safe filename based on the original audio file
                    safe_name = os.path.splitext(filename)[0]
                    debug_filepath = f"{safe_name}_translated.txt"
                    
                    try:
                        with open(debug_filepath, "w", encoding="utf-8") as text_file:
                            text_file.write(f"Original Language: {detected_lang}\n\n")
                            text_file.write(translated_lyrics)
                        print(f"  > [Debug] Translated lyrics saved to: {debug_filepath}")
                    except Exception as io_err:
                        print(f"  > [Debug] Failed to save lyrics .txt file: {io_err}")
                    # -------------------------------------------------------------------

                except Exception as e:
                    print(f"  > Translation Error: {e}")
        else:
            detected_lang = "none"

        import concurrent.futures

        def process_text():
            print(f"  > Processing text (Source: {lyrics_source})...")
            lyrics_clean = re.sub(r"\s+", " ", str(lyrics_text)).strip()
            full_tokens = self.tokenizer(lyrics_clean, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
            if full_tokens.dim() == 0:
                full_tokens = full_tokens.unsqueeze(0)

            chunk_ids_list, attn_mask_list = [], []
            for i in range(0, max(1, len(full_tokens)), config.TEXT_HOP_TOKENS):
                chunk_tokens = full_tokens[i : i + config.TEXT_WINDOW_TOKENS]
                chunk_ids = torch.cat([
                    torch.tensor([self.tokenizer.cls_token_id]),
                    chunk_tokens,
                    torch.tensor([self.tokenizer.sep_token_id]),
                ])
                if len(chunk_ids) < config.MAX_TEXT_LENGTH:
                    pad_len = config.MAX_TEXT_LENGTH - len(chunk_ids)
                    chunk_ids = torch.cat([chunk_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
                else:
                    chunk_ids = chunk_ids[: config.MAX_TEXT_LENGTH]
                chunk_ids_list.append(chunk_ids)
                attn_mask_list.append((chunk_ids != self.tokenizer.pad_token_id).long())

            text_loader = DataLoader(
                TensorDataset(torch.stack(chunk_ids_list), torch.stack(attn_mask_list)),
                batch_size=config.TEXT_BATCH_SIZE,
            )
            text_logits = []
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            ):
                for batch_ids, batch_masks in text_loader:
                    logits = self.text_model(
                        input_ids=batch_ids.to(self.device),
                        attention_mask=batch_masks.to(self.device),
                    ).logits
                    text_logits.extend(logits.float().cpu().numpy())
            return text_logits

        def process_audio():
            print(f"  > Processing audio...")
            v_raw, a_raw = self._get_valence_arousal(audio_path)
            audio_failed = False
            try:
                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != config.AUDIO_SAMPLE_RATE:
                    resampler = T.Resample(orig_freq=sr, new_freq=config.AUDIO_SAMPLE_RATE)
                    waveform = resampler(waveform)
                arr = waveform.squeeze().numpy()
            except Exception:
                arr = np.array([])
                audio_failed = True

            audio_logits = []
            if not audio_failed and len(arr) > 0:
                window_samples = int(config.AUDIO_WINDOW_SEC * config.AUDIO_SAMPLE_RATE)
                hop_samples = int(config.AUDIO_HOP_SEC * config.AUDIO_SAMPLE_RATE)
                audio_chunks = []
                for i in range(0, max(1, len(arr) - window_samples + 1), hop_samples):
                    chunk_arr = arr[i : i + window_samples]
                    if len(chunk_arr) < window_samples:
                        chunk_arr = np.pad(chunk_arr, (0, window_samples - len(chunk_arr)))
                    audio_chunks.append(chunk_arr)

                audio_loader = DataLoader(
                    TensorDataset(torch.tensor(np.array(audio_chunks))),
                    batch_size=config.AUDIO_BATCH_SIZE,
                )
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type,
                    dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
                ):
                    for batch in audio_loader:
                        batch_arrays = [b.numpy() for b in batch[0]]
                        current_bs = len(batch_arrays)
                        va_tensor = torch.tensor([[v_raw, a_raw]], dtype=torch.float32).repeat(current_bs, 1)
                        audio_inputs = self.feature_extractor(
                            batch_arrays, sampling_rate=config.AUDIO_SAMPLE_RATE, return_tensors="pt"
                        )
                        logits = self.audio_model(
                            audio_inputs["input_values"].to(self.device), va_tensor.to(self.device)
                        )
                        audio_logits.extend(logits.float().cpu().numpy())
            return audio_logits, audio_failed, v_raw, a_raw

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_text = executor.submit(process_text)
            future_audio = executor.submit(process_audio)
            text_logits_list = future_text.result()
            audio_logits_list, audio_failed, v_raw, a_raw = future_audio.result()

        print(f"  > Fusing multimodal features...")
        text_agg_logits = self.aggregate_confidence_weighted_chunks(text_logits_list)
        text_probs = self._softmax(text_agg_logits)
        text_conf = np.max(text_probs)

        audio_agg_logits = np.zeros(self.num_classes)
        audio_probs = np.zeros(self.num_classes)
        audio_conf = 0.0

        if not audio_failed and len(audio_logits_list) > 0:
            audio_agg_logits = self.aggregate_confidence_weighted_chunks(audio_logits_list)
            audio_probs = self._softmax(audio_agg_logits)
            audio_conf = np.max(audio_probs)

        if lyrics_source == "None" or text_conf < 0.25:
            final_probs = audio_probs if not audio_failed else text_probs
        elif audio_failed or audio_conf < 0.35:
            final_probs = text_probs
        else:
            n_text = self._normalize_logits(text_agg_logits)
            n_audio = self._normalize_logits(audio_agg_logits)
            f_logits = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                class_name = self.id_to_label[i].lower()
                w = config.MANUAL_WEIGHTS.get(class_name, config.MANUAL_WEIGHTS["default"])
                dt, da = w["text"] * text_conf, w["audio"] * audio_conf
                total_weight = dt + da + 1e-9
                f_logits[i] = (n_text[i] * (dt / total_weight)) + (n_audio[i] * (da / total_weight))
            final_probs = self._softmax(f_logits)

        text_idx = np.argmax(text_probs)
        text_mood = self.id_to_label[text_idx]
        text_conf = float(text_probs[text_idx])

        audio_mood, audio_conf = None, 0.0
        if not audio_failed and len(audio_probs) > 0:
            audio_idx = np.argmax(audio_probs)
            audio_mood = self.id_to_label[audio_idx]
            audio_conf = float(audio_probs[audio_idx])

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "Results": {
                "Predicted Mood": self.id_to_label[np.argmax(final_probs)],
                "Confidence": float(np.max(final_probs)),
            },
            "Individual": {
                "lyrics": {
                    "mood": text_mood,
                    "confidence": text_conf,
                    "text": lyrics_text,
                    "original_text": original_lyrics,
                    "translated_text": translated_lyrics,
                    "language": detected_lang,
                    "source": lyrics_source,
                },
                "audio": {"mood": audio_mood, "confidence": audio_conf},
            },
            "VA": {"valence": float(v_raw), "arousal": float(a_raw)},
        }

    def _softmax(self, logits):
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / np.sum(exp_l)

def extract_metadata(file_path):
    """
    Extract title and artist from ID3 tags, falling back to filename parsing.
    Handles 'Artist - Title' format in filenames.
    """
    stem = Path(file_path).stem
    try:
        audio = EasyID3(file_path)
        title = audio.get("title", [stem])[0]
        artist = audio.get("artist", ["Unknown Artist"])[0]

        # Fix: If artist is unknown but filename has a separator, trust the filename more
        if artist == "Unknown Artist" and " - " in stem:
            parts = stem.split(" - ", 1)
            artist, title = parts[0].strip(), parts[1].strip()
        
        return title, artist
    except Exception:
        # Fallback to filename parsing
        if " - " in stem:
            parts = stem.split(" - ", 1)
            return parts[1].strip(), parts[0].strip()
        return stem, "Unknown Artist"

def get_image_transform():
    import torchvision.transforms as transforms
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
