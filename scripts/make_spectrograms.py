# scripts/make_spectrograms.py

from sleepkit.logging_utils import setup_logger
from sleepkit.utils import (
    load_config, extract_labels, process_epoch_signal, save_epoch_spectrogram
)
import os
import pyedflib
import numpy as np

class SleepSpectrogramBuilder:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.raw_dir = self.config["raw_dir"]
        self.scoring_dir = self.config["scoring_dir"]
        self.out_dir = self.config["out_dir"]
        self.sr = self.config.get("sr", 256)
        self.nfft = self.config.get("nfft", 512)
        self.max_hz = self.config.get("max_hz", 64)
        self.channels = self.config["channels_of_interest"]
        self.channel_map = self.config["channel_map"]
        self.logger = setup_logger()

    def process_signals(self, raw_path):
        """Load and return dict of {channel_name: signal_array}."""
        try:
            signals_arr, signal_headers, _ = pyedflib.highlevel.read_edf(raw_path)
        except Exception as e:
            self.logger.error(f"Failed to read EDF: {raw_path}: {e}")
            return {}

        channel_labels = [h['label'] for h in signal_headers]
        self.logger.info(f"Channels in EDF: {channel_labels}")

        signals = {}
        for ch_idx in self.channels:
            ch_name = self.channel_map.get(ch_idx)
            if not ch_name:
                self.logger.warning(f"Channel index {ch_idx} not in channel_map.")
                continue
            if ch_name in channel_labels:
                signal_i = channel_labels.index(ch_name)
                sig = signals_arr[signal_i]
                signals[ch_name] = sig
                self.logger.info(f"Loaded signal '{ch_name}' (idx {ch_idx}), shape: {sig.shape}")
            else:
                self.logger.warning(f"Channel '{ch_name}' (idx {ch_idx}) not in EDF.")
        return signals

    def load_annotations(self, scoring_file):
        """Extract annotation events from scoring EDF."""
        if not os.path.exists(scoring_file):
            self.logger.warning(f"No scoring file: {scoring_file}")
            return []
        try:
            _, _, header_score = pyedflib.highlevel.read_edf(scoring_file)
            return header_score.get('annotations', [])
        except Exception as e:
            self.logger.error(f"Failed to read scoring EDF: {scoring_file}: {e}")
            return []

    def process_patient(self, patient_id, raw_path, scoring_path):
        """Process all channels and epochs for one patient."""
        self.logger.info(f"Processing patient {patient_id}")
        signals = self.process_signals(raw_path)
        if not signals:
            self.logger.error(f"No valid signals for {patient_id}, skipping.")
            return

        sample_length = min(len(s) for s in signals.values())
        n_epochs = sample_length // (30 * self.sr)
        self.logger.info(f"{patient_id}: {n_epochs} epochs (min channel length)")

        annotations = self.load_annotations(scoring_path)
        labels = extract_labels(annotations, n_epochs)

        for ch_name, sig in signals.items():
            for epoch_idx in range(n_epochs):
                start = epoch_idx * 30 * self.sr
                end = start + 30 * self.sr
                epoch_sig = sig[start:end]
                if len(epoch_sig) < 30 * self.sr:
                    self.logger.warning(f"Short epoch {epoch_idx} for {ch_name}, skipping.")
                    continue
                spec = process_epoch_signal(epoch_sig, self.sr, self.nfft, self.max_hz)
                save_epoch_spectrogram(
                    spec, self.out_dir, patient_id, epoch_idx, ch_name.replace(" ", "_")
                )
        self.logger.info(f"Finished {patient_id}")

    def run(self):
        """Main pipeline: processes all patients found in raw_dir."""
        patient_files = sorted(os.listdir(self.raw_dir))
        self.logger.info(f"Found {len(patient_files)} patient(s) in {self.raw_dir}")

        for pf in patient_files:
            patient_id = os.path.splitext(pf)[0]
            raw_path = os.path.join(self.raw_dir, pf)
            scoring_path = os.path.join(self.scoring_dir, pf)
            self.process_patient(patient_id, raw_path, scoring_path)
        self.logger.info("Done processing all patients.")

if __name__ == "__main__":
    builder = SleepSpectrogramBuilder()
    builder.run()