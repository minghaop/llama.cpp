#!/usr/bin/env python3
"""Compile a TFLite model to Qualcomm NPU AOT models with LiteRT."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors import import_vendor as _import_vendor
from ai_edge_litert.aot.vendors.qualcomm import qualcomm_backend as _qnn_backend  # noqa: F401
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


def _patch_import_vendor_aliases() -> None:
    """Allow common Qualcomm backend-id aliases across package variants."""
    orig_import_vendor = _import_vendor.import_vendor

    def import_vendor_compat(backend_id: str):
        aliases = {
            "QUALCOMM": ("qualcomm", "QNN"),
            "qualcomm": ("QUALCOMM", "QNN"),
            "QNN": ("QUALCOMM", "qualcomm"),
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


def _supported_soc_names() -> list[str]:
    names: list[str] = []
    for name, value in qnn_target.SocModel.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, qnn_target.SocModel):
            names.append(name)
    return sorted(names)


def parse_args() -> argparse.Namespace:
    soc_help = ", ".join(_supported_soc_names())

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=False, help="Input .tflite model path")
    p.add_argument("--out-dir", required=False, help="Output directory")
    p.add_argument("--model-name", default="flow", help="Output model base name")
    p.add_argument(
        "--soc",
        default="SM8750",
        help=(
            "Target SoC(s). Use one value (e.g. SM8750), a comma list "
            "(e.g. SM8750,SM8650), or ALL. Supported by current package: "
            f"{soc_help}"
        ),
    )
    p.add_argument("--keep-going", action="store_true", help="Keep compiling other targets when one fails")
    p.add_argument("--list-soc", action="store_true", help="List supported SoCs and exit")
    return p.parse_args()


def resolve_targets(soc_spec: str):
    normalized = soc_spec.strip().upper()
    if normalized == "ALL":
        return None

    supported = _supported_soc_names()
    if not supported:
        raise RuntimeError("No Qualcomm SoC model was found in ai_edge_litert package.")

    lookup = {name.upper(): name for name in supported}
    resolved = []

    for raw in soc_spec.split(","):
        token = raw.strip().upper().replace("-", "_")
        if not token:
            continue
        if token.isdigit():
            token = f"SM{token}"
        if token in lookup:
            resolved.append(qnn_target.Target(getattr(qnn_target.SocModel, lookup[token])))
            continue
        raise ValueError(
            f"Unsupported SoC '{raw}'. Supported: {', '.join(supported)} (or ALL)."
        )

    if not resolved:
        raise ValueError("No valid SoC was provided.")
    return resolved


def main() -> int:
    args = parse_args()
    _patch_import_vendor_aliases()

    if args.list_soc:
        print("Supported Qualcomm SoCs in current ai_edge_litert package:")
        for name in _supported_soc_names():
            print(f"- {name}")
        return 0

    if not args.model or not args.out_dir:
        raise SystemExit("ERROR: --model and --out-dir are required unless --list-soc is used.")

    model = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")

    targets = resolve_targets(args.soc)

    try:
        compiled_models = aot_lib.aot_compile(
            str(model),
            target=targets,
            keep_going=args.keep_going,
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
    report_path = out_dir / f"{args.model_name}_aot_report.txt"
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
    # Avoid accidental Python hash seed variability in downstream tools that parse logs.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
