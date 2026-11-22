"""Download every model the pipeline uses, at image build time.

Model names are read from the deployed config.yaml, so they cannot drift from
what scoring actually loads.
"""

from pathlib import Path

import yaml

cfg = yaml.safe_load(Path("config.yaml").read_text())

from faster_whisper import WhisperModel

WhisperModel(cfg["asr"]["model"], device="cpu", compute_type=cfg["asr"].get("compute_type", "int8"))

from sentence_transformers import SentenceTransformer

for name in {cfg["scoring"]["redundancy"]["embedding_model"],
             cfg["scoring"]["coherence"]["off_topic"]["embedding_model"]}:
    SentenceTransformer(name)

from transformers import ASTForAudioClassification, AutoFeatureExtractor

ast_name = cfg["scoring"]["coherence"]["background_music"]["model"]
AutoFeatureExtractor.from_pretrained(ast_name)
ASTForAudioClassification.from_pretrained(ast_name)

import spacy

spacy.load(cfg["onscreen"]["spacy_model"])

from rapidocr_onnxruntime import RapidOCR

RapidOCR()

import wordfreq

wordfreq.zipf_frequency("lecture", "en")
print("warmup: all models ready")
