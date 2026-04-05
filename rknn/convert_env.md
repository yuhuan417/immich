# RKNN conversion environment

This repo uses a dedicated Python virtualenv for ONNX -> RKNN conversion.
The setup has been verified with:

- Python 3.11
- `rknn-toolkit2==2.3.2`
- `target_platform=rk3576`

## Quick start

Create or refresh the conversion environment:

```bash
./tools/setup_convert_env.sh
```

If `rknn-toolkit2` must be installed from a local wheel, use:

```bash
./tools/setup_convert_env.sh --rknn-wheel /path/to/rknn_toolkit2-2.3.2-*.whl
```

The script creates `<repo>/.venv`, installs the pinned dependencies from
`tools/requirements-convert.txt`, installs `rknn-toolkit2==2.3.2`, and checks
that `from rknn.api import RKNN` works.

## Run conversion

Use the virtualenv Python directly:

```bash
./.venv/bin/python ./tools/convert_ppocrv5_rknn.py --model all --target-platform rk3576
```

For another chip later, only change the target:

```bash
./.venv/bin/python ./tools/convert_ppocrv5_rknn.py --model all --target-platform rk3588
```

## Output files

The converter writes:

- `detection/rknpu/<target_platform>/model.rknn`
- `recognition/rknpu/<target_platform>/model.rknn`
- `output/rknn_conversion_report.txt`

## Notes

- The conversion contract keeps preprocessing outside RKNN. Do not add
  `mean_values` or `std_values` in `rknn.config()`.
- The script does not change ONNX input/output tensor names.
- `--report-only` is useful for re-generating SHA256 and metadata from
  existing RKNN files without rebuilding them.
