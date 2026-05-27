#!/usr/bin/env python3
"""Compile flow.tflite to Google Tensor NPU AOT models.

Requires Linux x86_64 with AVX support.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors.google_tensor import target as gt_target
from ai_edge_litert.aot.vendors.google_tensor import google_tensor_backend as _gt_backend  # noqa: F401
from ai_edge_litert.aot.vendors import import_vendor as _import_vendor


def _patch_import_vendor_aliases() -> None:
    """Allow common Google Tensor backend-id aliases across package variants."""
    orig_import_vendor = _import_vendor.import_vendor

    def import_vendor_compat(backend_id: str):
        aliases = {
            "GOOGLE": ("GOOGLE_TENSOR", "google_tensor", "Google_Tensor"),
            "GOOGLE_TENSOR": ("GOOGLE", "google_tensor", "Google_Tensor"),
            "google_tensor": ("GOOGLE_TENSOR", "GOOGLE", "Google_Tensor"),
            "Google_Tensor": ("GOOGLE_TENSOR", "GOOGLE", "google_tensor"),
        }
        tried = []
        for candidate in (backend_id, *aliases.get(backend_id, ())):
            if candidate in tried:
                continue
            tried.append(candidate)
            try:
                return orig_import_vendor(candidate)
            except ValueError:
                continue
        return orig_import_vendor(backend_id)

    _import_vendor.import_vendor = import_vendor_compat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Input .tflite model path")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--model-name", default="flow", help="Output model base name")
    p.add_argument(
        "--soc",
        default="G5",
        choices=["G3", "G4", "G5", "ALL"],
        help="Google Tensor target SoC",
    )
    p.add_argument("--keep-going", action="store_true", help="Keep compiling other targets when one fails")
    p.add_argument(
        "--enable-large-model-support",
        action="store_true",
        help="Enable Google Tensor large-model compilation support",
    )
    p.add_argument(
        "--truncation-type",
        default=None,
        choices=["auto", "none", "bfloat16", "half"],
        help="Google Tensor float truncation type",
    )
    p.add_argument(
        "--sharding-intensity",
        default=None,
        choices=["minimal", "moderate", "extensive", "maximum"],
        help="Google Tensor sharding intensity",
    )
    return p.parse_args()


def resolve_targets(soc: str):
    if soc == "ALL":
        return None
    mapping = {
        "G3": gt_target.SocModel.TENSOR_G3,
        "G4": gt_target.SocModel.TENSOR_G4,
        "G5": gt_target.SocModel.TENSOR_G5,
    }
    return [gt_target.Target(mapping[soc])]


def main() -> int:
    args = parse_args()
    _patch_import_vendor_aliases()
    model = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sdk_tar = os.environ.get("GOOGLE_TENSOR_SDK_BETA", "")
    if not sdk_tar:
        raise RuntimeError("GOOGLE_TENSOR_SDK_BETA is not set (must point to litert_plugin_compiler.tar.gz)")

    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")

    targets = resolve_targets(args.soc)
    compile_kwargs: dict[str, object] = {}
    if args.enable_large_model_support:
        compile_kwargs["google_tensor_enable_large_model_support"] = True
    if args.truncation_type:
        mapping = {
            "auto": "auto",
            "none": "no_truncation",
            "bfloat16": "bfloat16",
            "half": "half",
        }
        compile_kwargs["google_tensor_truncation_type"] = mapping[args.truncation_type]
    if args.sharding_intensity:
        compile_kwargs["google_tensor_sharding_intensity"] = args.sharding_intensity

    try:
        compiled_models = aot_lib.aot_compile(
            str(model),
            target=targets,
            keep_going=args.keep_going,
            **compile_kwargs,
        )
    except Exception as exc:
        msg = str(exc)
        print("AOT compile failed:", msg)
        m = re.search(r"(/tmp/\S+\.error)", msg)
        if m:
            err_path = Path(m.group(1))
            print(f"Detected error log: {err_path}")
            if err_path.is_file():
                try:
                    text = err_path.read_text(errors="replace")
                    lines = text.splitlines()
                    print("=== ERROR LOG (last 120 lines) ===")
                    for line in lines[-120:]:
                        print(line)
                    print("=== KEY LINES ===")
                    key_re = re.compile(
                        r"(Failed to compile model|INTERNAL|FATAL|Check failed|OOM|"
                        r"bad_alloc|Aborted|Segmentation|compiler_worker|ERROR:)"
                    )
                    for line in lines:
                        if key_re.search(line):
                            print(line)
                except Exception as log_exc:
                    print(f"Failed to read error log: {log_exc}")
        raise

    report = compiled_models.compilation_report()
    report_path = out_dir / "flow_aot_report.txt"
    report_path.write_text(report)

    compiled_models.export(str(out_dir), model_name=args.model_name)

    print("=== AOT REPORT ===")
    print(report)
    print("=== EXPORTED FILES ===")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
