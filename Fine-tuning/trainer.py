import copy
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
import lightning.pytorch as pl
from omegaconf import OmegaConf

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.preprocessing.perturb import (
    AudioAugmentor,
    GainPerturbation,
    SpeedPerturbation,
    TimeStretchPerturbation,
    WhiteNoisePerturbation,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class ModelConfig:
    # --- model ---
    name: str
    loader: str  # "conformer" or "quartznet"

    # --- paths ---
    base_dir: str
    collected_dataset_path: str
    synthesized_dataset_path: str
    output_dir: str

    # optional / legacy (kept for compatibility)
    processed_dataset_path: str = ""
    healthy_dataset_path: str = ""
    pathological_dataset_path: str = ""
    easycall_dataset_path: str = ""

    # --- audio / data ---
    sample_rate: int = 16000
    normalize_text: bool = True
    force_reprocess: bool = False

    # --- training ---
    batch_size: int = 16
    max_train_duration: float = 10.0
    max_val_duration: float = 20.0
    num_workers: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    min_epochs: int = 40
    max_epochs: int = 60
    patience: int = 20
    precision: int | str = 32
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 300
    seed: int = 42

    # --- augmentation ---
    use_augmentation: bool = False
    perturb: dict | None = None  # optional YAML overrides

    # --- weighting (oversampling collected data) ---
    weight_factor: int = 40


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
class BaseDatasetBuilder:
    """Base helper to build NeMo-compatible JSON manifests."""

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.base_dir = Path(config.base_dir)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()

    def _maybe_normalize(self, text: str) -> str:
        return self._normalize_text(text) if self.cfg.normalize_text else text

    def _duration(self, audio_path: str | Path) -> float:
        return AudioSegment.from_file(
            str(audio_path), target_sr=self.cfg.sample_rate
        ).duration

    def _write_manifest(self, rows: List[Dict[str, Any]], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def make_manifest(self, items: List[Dict[str, Any]], out_path: Path) -> None:
        """Compute durations + normalize text and write JSONL manifest."""
        processed: List[Dict[str, Any]] = []

        for it in items:
            ap = Path(it["audio_filepath"])
            txt = self._maybe_normalize(str(it["text"]))
            try:
                dur = self._duration(ap)
            except Exception as e:
                logging.warning("Skipping file due to error: %s (%s)", ap, e)
                continue

            if dur <= 0:
                logging.warning(
                    "Skipping file with non-positive duration: %s", ap
                )
                continue

            processed.append(
                {"audio_filepath": str(ap), "duration": float(dur), "text": txt}
            )

        self._write_manifest(processed, out_path)

    def build(self) -> Dict[str, str]:
        raise NotImplementedError


class SynthesizedDatasetBuilder(BaseDatasetBuilder):
    """Create train/eval manifests from synthesized CSV."""

    def __init__(
        self, config: ModelConfig, csv_name: str = "generated_metadata_filtered.csv"
    ):
        super().__init__(config)
        self.csv_path = Path(config.synthesized_dataset_path) / csv_name

    def _pick_col(self, df: pd.DataFrame, candidates: List[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"No column in {candidates} found in {self.csv_path.name}"
        )

    def build(self) -> Dict[str, str]:
        out_dir = Path(self.cfg.output_dir)
        train_manifest = out_dir / "synth_train.json"
        eval_manifest = out_dir / "synth_eval.json"

        df = pd.read_csv(self.csv_path)
        audio_col = self._pick_col(
            df, ["audio_path", "filename_path", "audio_filepath"]
        )
        text_col = self._pick_col(df, ["generated_text", "sentence", "text"])

        records = [
            {"audio_filepath": str(r[audio_col]), "text": str(r[text_col])}
            for _, r in df.iterrows()
        ]

        random.shuffle(records)
        n_eval = max(1, len(records) // 7)  # ≈14% eval
        eval_recs = records[:n_eval]
        train_recs = records[n_eval:]

        self.make_manifest(train_recs, train_manifest)
        self.make_manifest(eval_recs, eval_manifest)

        return {"train": str(train_manifest), "eval": str(eval_manifest)}


class CollectedDatasetBuilder(BaseDatasetBuilder):
    """
    Build manifests from collected dataset CSV (dataset.csv).
    Splits by 'id_sess' so full sessions stay in train or eval.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.root = Path(config.collected_dataset_path)
        df = pd.read_csv(self.root / "dataset.csv")

        def is_true(v) -> bool:
            if pd.isna(v):
                return False
            return str(v).strip().lower() in {"t", "true", "1", "yes"}

        df["is_health_norm"] = df["is_health"].apply(is_true)
        self.healthy_records = df[df["is_health_norm"]].to_dict(orient="records")
        self.pathological_records = df[~df["is_health_norm"]].to_dict(
            orient="records"
        )

    @staticmethod
    def _group_by_session(
        records: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        sessions: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            sid = rec.get("id_sess", "unknown")
            sessions.setdefault(str(sid), []).append(rec)
        return list(sessions.values())

    def _preprocess(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "audio_filepath": str(Path(self.root) / str(rec["filename_path"])),
            "text": str(rec["text"]),
        }

    def build(self) -> Dict[str, str]:
        out_dir = Path(self.cfg.output_dir)
        paths = {
            "train": out_dir / "collected_train.json",
            "eval": out_dir / "collected_eval.json",
            "healthy_eval": out_dir / "collected_healthy_eval.json",
            "pathological_eval": out_dir / "collected_pathological_eval.json",
        }

        healthy_sessions = self._group_by_session(self.healthy_records)
        pathological_sessions = self._group_by_session(
            self.pathological_records
        )

        random.shuffle(healthy_sessions)
        random.shuffle(pathological_sessions)

        n_train_healthy = len(healthy_sessions) // 2
        n_train_path = len(pathological_sessions) // 2

        train_sessions = (
            healthy_sessions[:n_train_healthy] + pathological_sessions[:n_train_path]
        )
        eval_sessions = (
            healthy_sessions[n_train_healthy:] + pathological_sessions[n_train_path:]
        )

        train_records = [r for s in train_sessions for r in s]
        eval_records = [r for s in eval_sessions for r in s]

        healthy_eval = [r for r in eval_records if r in self.healthy_records]
        pathological_eval = [
            r for r in eval_records if r in self.pathological_records
        ]

        # Weighted train: duplicate each collected example N times
        if not paths["train"].exists() or self.cfg.force_reprocess:
            raw = [self._preprocess(r) for r in train_records]
            weighted: List[Dict[str, Any]] = []
            for rec in raw:
                weighted.extend([rec] * self.cfg.weight_factor)

            print(
                f"Writing train manifest: {len(weighted)} lines "
                f"(factor {self.cfg.weight_factor})"
            )
            self.make_manifest(weighted, paths["train"])

        # Eval (no duplication)
        if not paths["eval"].exists() or self.cfg.force_reprocess:
            self.make_manifest(
                [self._preprocess(r) for r in eval_records], paths["eval"]
            )

        if not paths["healthy_eval"].exists() or self.cfg.force_reprocess:
            self.make_manifest(
                [self._preprocess(r) for r in healthy_eval], paths["healthy_eval"]
            )

        if not paths["pathological_eval"].exists() or self.cfg.force_reprocess:
            self.make_manifest(
                [self._preprocess(r) for r in pathological_eval],
                paths["pathological_eval"],
            )

        return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
class ModelTrainer:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        # build manifests
        self.collected = CollectedDatasetBuilder(cfg).build()
        self.synth = SynthesizedDatasetBuilder(cfg).build()

    def _make_augmentor(self) -> AudioAugmentor | None:
        if not self.cfg.use_augmentation:
            return None

        defaults = {
            "speed": {
                "enabled": True,
                "min": 0.9,
                "max": 1.1,
                "resample_type": "kaiser_fast",
            },
            "white_noise": {"enabled": True, "min_level": -90, "max_level": -46},
            "time_stretch": {"enabled": True, "min": 0.8, "max": 1.2, "num_rates": 3},
            "gain": {"enabled": True, "min_db": -12, "max_db": 12},
        }

        user = self.cfg.perturb or {}
        for k, v in user.items():
            if k in defaults and isinstance(v, dict):
                defaults[k].update(v)

        perts = []
        if defaults["speed"]["enabled"]:
            perts.append(
                SpeedPerturbation(
                    sr=self.cfg.sample_rate,
                    resample_type=defaults["speed"]["resample_type"],
                    min_speed_rate=defaults["speed"]["min"],
                    max_speed_rate=defaults["speed"]["max"],
                )
            )
        if defaults["white_noise"]["enabled"]:
            perts.append(
                WhiteNoisePerturbation(
                    min_level=defaults["white_noise"]["min_level"],
                    max_level=defaults["white_noise"]["max_level"],
                    rng=None,
                )
            )
        if defaults["time_stretch"]["enabled"]:
            perts.append(
                TimeStretchPerturbation(
                    min_speed_rate=defaults["time_stretch"]["min"],
                    max_speed_rate=defaults["time_stretch"]["max"],
                    num_rates=defaults["time_stretch"]["num_rates"],
                    rng=None,
                )
            )
        if defaults["gain"]["enabled"]:
            perts.append(
                GainPerturbation(
                    min_gain_dbfs=defaults["gain"]["min_db"],
                    max_gain_dbfs=defaults["gain"]["max_db"],
                    rng=None,
                )
            )

        return AudioAugmentor(perturbations=perts) if perts else None

    def _load_model(self):
        cls = {
            "quartznet": nemo_asr.models.EncDecCTCModel,
            "conformer": nemo_asr.models.EncDecCTCModelBPE,
        }.get(self.cfg.loader, nemo_asr.models.EncDecCTCModel)

        model = cls.from_pretrained(self.cfg.name, map_location="cpu")

        # dataset configs
        train_cfg = copy.deepcopy(model.cfg.train_ds)
        val_cfg = copy.deepcopy(model.cfg.validation_ds)

        train_cfg.manifest_filepath = [
            self.collected["train"],
            self.synth["train"],
        ]
        val_cfg.manifest_filepath = [
            self.collected["eval"],
            self.collected["healthy_eval"],
            self.collected["pathological_eval"],
        ]

        for ds in (train_cfg, val_cfg):
            ds.sample_rate = self.cfg.sample_rate
            ds.num_workers = self.cfg.num_workers
            ds.pin_memory = True

        # training overrides
        train_cfg.batch_size = self.cfg.batch_size
        train_cfg.shuffle = True
        OmegaConf.set_struct(train_cfg, False)
        train_cfg.max_duration = float(self.cfg.max_train_duration)
        train_cfg.is_tarred = False
        OmegaConf.set_struct(train_cfg, True)

        # validation overrides
        val_cfg.batch_size = self.cfg.batch_size
        val_cfg.shuffle = False
        OmegaConf.set_struct(val_cfg, False)
        val_cfg.max_duration = float(self.cfg.max_val_duration)
        OmegaConf.set_struct(val_cfg, True)

        model.cfg.train_ds = train_cfg
        model.cfg.validation_ds = val_cfg

        # optimizer
        model.cfg.optim.lr = float(self.cfg.learning_rate)
        model.cfg.optim.weight_decay = float(self.cfg.weight_decay)

        # dataloaders
        model.setup_training_data(train_cfg)
        model.setup_multiple_validation_data(val_cfg)

        # optional augmentation
        aug = self._make_augmentor()
        if aug and hasattr(model, "_train_dl") and model._train_dl is not None:
            model._train_dl.dataset.augmentor = aug

        return model

    def train(self) -> None:
        model = self._load_model()

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        log_dir = os.path.join(self.cfg.output_dir, f"logs/job_{job_id}")
        os.makedirs(log_dir, exist_ok=True)

        ckpt_dir = os.path.join(self.cfg.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        trainer = pl.Trainer(
            num_sanity_val_steps=2,
            min_epochs=self.cfg.min_epochs,
            max_epochs=self.cfg.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=self.cfg.precision,
            gradient_clip_val=self.cfg.gradient_clip_val,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename="{epoch:02d}-{val_wer:.2f}",
                    save_top_k=1,
                    monitor="val_wer",
                    mode="min",
                    save_last=True,
                ),
                pl.callbacks.EarlyStopping(
                    monitor="val_wer", patience=self.cfg.patience, mode="min"
                ),
                pl.callbacks.LearningRateMonitor(logging_interval="step"),
            ],
            logger=[pl.loggers.CSVLogger(save_dir=log_dir)],
            log_every_n_steps=self.cfg.log_every_n_steps,
        )

        model.set_trainer(trainer)
        trainer.validate(model)
        trainer.fit(model)

        final_path = os.path.join(self.cfg.output_dir, "final_model.nemo")
        model.save_to(final_path)
        print("Training finished.")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=logging.INFO,
    )


def main() -> None:
    setup_logging()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    mc = ModelConfig(
        name=str(cfg_yaml["name"]),
        loader=str(cfg_yaml["loader"]).lower(),
        base_dir=str(cfg_yaml["paths"]["base_dir"]),
        collected_dataset_path=str(cfg_yaml["paths"]["collected_dataset"]),
        synthesized_dataset_path=str(cfg_yaml["paths"]["synthesized_dataset"]),
        output_dir=str(cfg_yaml["paths"]["output_dir"]),
        processed_dataset_path=str(cfg_yaml["paths"].get("processed_dataset", "")),
        healthy_dataset_path=str(cfg_yaml["paths"].get("healthy_dataset", "")),
        pathological_dataset_path=str(
            cfg_yaml["paths"].get("pathological_dataset", "")
        ),
        easycall_dataset_path=str(cfg_yaml["paths"].get("easycall_dataset", "")),
        sample_rate=int(cfg_yaml.get("sample_rate", 16000)),
        normalize_text=bool(cfg_yaml.get("normalize_text", True)),
        force_reprocess=bool(cfg_yaml.get("force_reprocess", False)),
        use_augmentation=bool(cfg_yaml.get("use_augmentation", False)),
        weight_factor=int(cfg_yaml.get("weight_factor", 40)),
        perturb=cfg_yaml.get("perturb", None),
        batch_size=int(cfg_yaml.get("batch_size, 16").split(",")[0])  # robust parse
        if isinstance(cfg_yaml.get("batch_size", 16), str)
        else int(cfg_yaml.get("batch_size", 16)),
        max_train_duration=float(cfg_yaml.get("max_train_duration", 10.0)),
        max_val_duration=float(cfg_yaml.get("max_val_duration", 20.0)),
        num_workers=int(cfg_yaml.get("num_workers", 8)),
        learning_rate=float(cfg_yaml.get("learning_rate", 1e-5)),
        weight_decay=float(cfg_yaml.get("weight_decay", 1e-3)),
        min_epochs=int(cfg_yaml.get("min_epochs", 40)),
        max_epochs=int(cfg_yaml.get("max_epochs", 60)),
        patience=int(cfg_yaml.get("patience", 20)),
        precision=cfg_yaml.get("precision", 32),
        gradient_clip_val=float(cfg_yaml.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(cfg_yaml.get("log_every_n_steps", 300)),
        seed=int(cfg_yaml.get("seed", 42)),
    )

    trainer = ModelTrainer(mc)
    trainer.train()


if __name__ == "__main__":
    main()
