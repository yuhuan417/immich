#!/usr/bin/env python3

import argparse
import hashlib
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PLATFORM = "rk3576"
DO_QUANTIZATION = False
EXPECTED_RKNN_TOOLKIT2_VERSION = "2.3.2"

MODELS = {
    "detection": {
        "source": ROOT / "PP-OCRv5_mobile_det" / "inference.onnx",
        "input_name": "x",
        "output_name": "fetch_name_0",
        "dtype": "float32",
        "layout": "NCHW",
        "color_order": "BGR",
        "dynamic_input": [
            [[1, 3, 736, 736]],
            [[1, 3, 736, 1280]],
            [[1, 3, 1280, 736]],
        ],
    },
    "recognition": {
        "source": ROOT / "PP-OCRv5_mobile_rec" / "inference.onnx",
        "input_name": "x",
        "output_name": "fetch_name_0",
        "output_last_dim": 18385,
        "dtype": "float32",
        "layout": "NCHW",
        "color_order": "BGR",
        "dynamic_input": [
            [[1, 3, 48, 320]],
            [[1, 3, 48, 640]],
            [[1, 3, 48, 960]],
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PP-OCRv5 ONNX models to RKNN.")
    parser.add_argument(
        "--model",
        choices=["all", *MODELS.keys()],
        default="all",
        help="Convert a single model or both models.",
    )
    parser.add_argument(
        "--target-platform",
        default=DEFAULT_TARGET_PLATFORM,
        help="RKNN target platform and output directory name.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable RKNN verbose logging.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only write the report using existing artifacts.",
    )
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_rknn_toolkit2_version() -> str | None:
    try:
        return version("rknn-toolkit2")
    except PackageNotFoundError:
        return None


def ensure_supported_toolkit_version() -> str:
    installed_version = get_rknn_toolkit2_version()
    if installed_version is None:
        raise RuntimeError(
            "rknn-toolkit2 is not installed. "
            f"Run tools/setup_convert_env.sh or install {EXPECTED_RKNN_TOOLKIT2_VERSION} before converting."
        )

    normalized_version = installed_version.split("+", 1)[0]
    if normalized_version != EXPECTED_RKNN_TOOLKIT2_VERSION:
        raise RuntimeError(
            "Unsupported rknn-toolkit2 version: "
            f"{installed_version}. Expected {EXPECTED_RKNN_TOOLKIT2_VERSION}."
        )

    return installed_version


def create_rknn(*, verbose: bool):
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import rknn.api. "
            f"Run tools/setup_convert_env.sh or install rknn-toolkit2 {EXPECTED_RKNN_TOOLKIT2_VERSION}."
        ) from exc

    return RKNN(verbose=verbose)


def artifact_path(model_name: str, target_platform: str) -> Path:
    return ROOT / model_name / "rknpu" / target_platform / "model.rknn"


def convert(model_name: str, *, target_platform: str, verbose: bool) -> Path:
    config = MODELS[model_name]
    source = config["source"]
    output = artifact_path(model_name, target_platform)
    output.parent.mkdir(parents=True, exist_ok=True)

    rknn = create_rknn(verbose=verbose)
    try:
        print(f"[{model_name}] source: {source}")
        print(f"[{model_name}] output: {output}")
        print(f"[{model_name}] target_platform: {target_platform}")
        print(f"[{model_name}] input_name: {config['input_name']}")
        print(f"[{model_name}] output_name: {config['output_name']}")
        print(f"[{model_name}] dynamic_input: {config['dynamic_input']}")

        ret = rknn.config(
            target_platform=target_platform,
            dynamic_input=config["dynamic_input"],
        )
        if ret != 0:
            raise RuntimeError(f"{model_name}: rknn.config failed with code {ret}")

        ret = rknn.load_onnx(model=str(source))
        if ret != 0:
            raise RuntimeError(f"{model_name}: rknn.load_onnx failed with code {ret}")

        ret = rknn.build(do_quantization=DO_QUANTIZATION)
        if ret != 0:
            raise RuntimeError(f"{model_name}: rknn.build failed with code {ret}")

        ret = rknn.export_rknn(str(output))
        if ret != 0:
            raise RuntimeError(f"{model_name}: rknn.export_rknn failed with code {ret}")

        return output
    finally:
        rknn.release()


def write_report(outputs: dict[str, Path], *, target_platform: str) -> Path:
    report_path = ROOT / "output" / "rknn_conversion_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    installed_version = get_rknn_toolkit2_version() or "not installed in current environment"

    lines = [
        f"rknn-toolkit2: {installed_version}",
        f"required_rknn_toolkit2: {EXPECTED_RKNN_TOOLKIT2_VERSION}",
        f"target_platform: {target_platform}",
        f"do_quantization: {DO_QUANTIZATION}",
        "mean_values/std_values: not set",
        "input_preprocess: external (Immich), not baked into RKNN",
        "",
    ]
    for model_name, artifact in outputs.items():
        lines.extend(
            [
                f"[{model_name}]",
                f"source: {MODELS[model_name]['source'].relative_to(ROOT)}",
                f"output: {artifact.relative_to(ROOT)}",
                f"input_name: {MODELS[model_name]['input_name']}",
                f"output_name: {MODELS[model_name]['output_name']}",
                f"dtype: {MODELS[model_name]['dtype']}",
                f"layout: {MODELS[model_name]['layout']}",
                f"color_order: {MODELS[model_name]['color_order']}",
                f"dynamic_input: {MODELS[model_name]['dynamic_input']}",
                (
                    f"output_last_dim: {MODELS[model_name]['output_last_dim']}"
                    if "output_last_dim" in MODELS[model_name]
                    else "output_last_dim: n/a"
                ),
                f"sha256: {sha256sum(artifact)}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def collect_outputs(model_names: list[str], *, target_platform: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for model_name in model_names:
        artifact = artifact_path(model_name, target_platform)
        if not artifact.exists():
            raise FileNotFoundError(f"Missing artifact for {model_name}: {artifact}")
        outputs[model_name] = artifact
    return outputs


def run_isolated(model_name: str, *, target_platform: str, verbose: bool) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        model_name,
        "--target-platform",
        target_platform,
    ]
    if verbose:
        command.append("--verbose")
    temp_dir = ROOT / "output" / "tmp" / model_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["TEMP"] = str(temp_dir)
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def main() -> int:
    args = parse_args()
    requested = list(MODELS) if args.model == "all" else [args.model]
    installed_version = get_rknn_toolkit2_version()

    print(f"rknn-toolkit2: {installed_version or 'not installed'}")
    if args.report_only:
        outputs = collect_outputs(requested, target_platform=args.target_platform)
    elif args.model == "all":
        ensure_supported_toolkit_version()
        for model_name in requested:
            run_isolated(model_name, target_platform=args.target_platform, verbose=args.verbose)
        outputs = collect_outputs(requested, target_platform=args.target_platform)
    else:
        ensure_supported_toolkit_version()
        outputs = {
            args.model: convert(args.model, target_platform=args.target_platform, verbose=args.verbose)
        }

    report_path = write_report(outputs, target_platform=args.target_platform)
    print(f"report: {report_path}")
    for model_name, artifact in outputs.items():
        print(f"{model_name}: {artifact} sha256={sha256sum(artifact)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
