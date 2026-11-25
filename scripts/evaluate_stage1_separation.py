"""
Evaluation Script for Stage 1: Blind Source Separation (Milestone M2).

Evaluates Demucs (primary) and Conv-TasNet (comparative baseline) on the
Controlled Evaluation Suite, calculating PIT SI-SDR, SDR, latency, and RTF.
Generates demonstration audio artifacts in outputs/first_eval_demo/.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bss_metrics import evaluate_separation_pit
from src.core.audio_io import load_audio
from src.separation.conv_tasnet_wrapper import ConvTasNetSeparator
from src.separation.demucs_wrapper import DemucsSeparator


def main():
    print("=" * 80)
    print("STAGE 1: BLIND SOURCE SEPARATION EVALUATION (MILESTONE M2)")
    print("=" * 80)

    manifest_file = Path("data/manifests/controlled_ground_truth.json")
    assert manifest_file.exists(), f"Missing manifest: {manifest_file}"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mixtures = [m for m in manifest["mixtures"] if m.get("num_speakers", 2) == 2 and "overlap_ratio_target" in m][:6]
    demo_out_dir = Path("outputs/first_eval_demo")
    demo_out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    device_pref = "auto"
    print(f"\nInitializing Demucs (htdemucs) on device '{device_pref}'...")
    demucs = DemucsSeparator(device=device_pref)
    demucs.initialize()
    print(f"Demucs initialized on: {demucs.device}")

    print(f"\nInitializing Conv-TasNet on device '{device_pref}'...")
    conv_tasnet = ConvTasNetSeparator(device=device_pref)
    conv_tasnet.initialize()
    print(f"Conv-TasNet initialized on: {conv_tasnet.device}")

    results = []

    print("\nExecuting Source Separation & PIT SI-SDR Evaluation...")
    print("-" * 80)

    for item in mixtures:
        mix_id = item["mixture_id"]
        mix_path = Path(item["mixture_file_path"])
        duration = item["total_duration_seconds"]

        # Aligned ground-truth stems
        gt_a_path = Path(item["sources"][0]["aligned_ground_truth_path"])
        gt_b_path = Path(item["sources"][1]["aligned_ground_truth_path"])

        tensor_mix, sr = load_audio(mix_path, target_sr=16000, to_mono=True)
        tensor_gt_a, _ = load_audio(gt_a_path, target_sr=16000, to_mono=True)
        tensor_gt_b, _ = load_audio(gt_b_path, target_sr=16000, to_mono=True)

        sig_mix = tensor_mix.squeeze(0).cpu().numpy()
        sig_gt_a = tensor_gt_a.squeeze(0).cpu().numpy()
        sig_gt_b = tensor_gt_b.squeeze(0).cpu().numpy()

        references = [sig_gt_a, sig_gt_b]

        # -------------------------------------------------------------
        # 1. Evaluate Demucs
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        stems_demucs, meta_demucs = demucs.separate_file(
            mix_path,
            output_dir=demo_out_dir / "demucs" / mix_id,
        )
        time_demucs = time.perf_counter() - t0
        rtf_demucs = round(time_demucs / duration, 3)

        # PIT SI-SDR evaluation for Demucs
        pit_demucs = evaluate_separation_pit(
            references=references,
            estimations=stems_demucs,
            mixture=sig_mix,
        )

        # -------------------------------------------------------------
        # 2. Evaluate Conv-TasNet (Baseline)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        stems_tasnet, meta_tasnet = conv_tasnet.separate_file(
            mix_path,
            output_dir=demo_out_dir / "conv_tasnet" / mix_id,
        )
        time_tasnet = time.perf_counter() - t0
        rtf_tasnet = round(time_tasnet / duration, 3)

        # PIT SI-SDR evaluation for Conv-TasNet
        pit_tasnet = evaluate_separation_pit(
            references=references,
            estimations=stems_tasnet,
            mixture=sig_mix,
        )

        record = {
            "mixture_id": mix_id,
            "duration_seconds": duration,
            "overlap_ratio": item["overlap_ratio_target"],
            "sir_db": item["sir_target_db"],
            "baseline_mixture_sisdr_db": pit_demucs["baseline_mixture_sisdr_db"],
            "mean_baseline_sisdr_db": pit_demucs["mean_baseline_sisdr_db"],
            "demucs": {
                "mean_sisdr_db": pit_demucs["mean_sisdr_db"],
                "source_sisdr_db": pit_demucs["source_sisdr_db"],
                "sisdr_improvement_delta_db": pit_demucs["sisdr_improvement_delta_db"],
                "mean_delta_sisdr_db": pit_demucs["mean_sisdr_improvement_delta_db"],
                "processing_time_seconds": round(time_demucs, 3),
                "rtf": rtf_demucs,
                "separated_stems": [s["file_path"] for s in meta_demucs["stems"]],
            },
            "conv_tasnet": {
                "mean_sisdr_db": pit_tasnet["mean_sisdr_db"],
                "source_sisdr_db": pit_tasnet["source_sisdr_db"],
                "sisdr_improvement_delta_db": pit_tasnet["sisdr_improvement_delta_db"],
                "mean_delta_sisdr_db": pit_tasnet["mean_sisdr_improvement_delta_db"],
                "processing_time_seconds": round(time_tasnet, 3),
                "rtf": rtf_tasnet,
                "separated_stems": [s["file_path"] for s in meta_tasnet["stems"]],
            },
        }
        results.append(record)

        print(f"[{mix_id}] Duration: {duration}s | Overlap: {int(item['overlap_ratio_target']*100)}% | SIR: {item['sir_target_db']}dB")
        print(f"  Input Baseline SI-SDR : {pit_demucs['mean_baseline_sisdr_db']} dB")
        print(f"  Demucs Output SI-SDR   : {pit_demucs['mean_sisdr_db']} dB (Δ = +{pit_demucs['mean_sisdr_improvement_delta_db']} dB) | RTF: {rtf_demucs}")
        print(f"  Conv-TasNet Output SDR : {pit_tasnet['mean_sisdr_db']} dB (Δ = +{pit_tasnet['mean_sisdr_improvement_delta_db']} dB) | RTF: {rtf_tasnet}")
        print("-" * 80)

    # Save summary report
    summary_file = demo_out_dir / "stage1_separation_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"total_evaluations": len(results), "results": results}, f, indent=2)

    print(f"\nDetailed evaluation results saved to: {summary_file}")


if __name__ == "__main__":
    main()
