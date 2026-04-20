# AI-Powered Sign Language Interpreter

This repository focuses on dataset creation and validation workflows for a Turkish Sign Language (TSL) interpreter project.

## Scope

Current implementation includes:

- Webcam-based landmark sequence collection (MediaPipe Tasks API)
- YouTube-based dataset generation pipeline (yt-dlp + MediaPipe)
- Dataset verification tools for `.npy` sequence integrity

This repository currently does not include full model training and real-time inference application code.

## Repository Structure

- `collect_data.py`: Captures webcam frames and saves sign sequences as `(30, 63)` arrays.
- `youtube_dataset_builder.py`: Downloads videos from a target channel and converts them to landmark sequences.
- `verify_data.py`: Validates dataset files (shape checks, NaN checks, file read checks).
- `_test_import.py`: Basic environment check (MediaPipe/OpenCV/Numpy/model file).
- `data/`: Dataset root directory (one class folder per sign label).
- `models/`: Stores model assets (for example `hand_landmarker.task`).
- `data_collection/`: Mirror copies of the same tooling scripts.

## Requirements

- Python 3.12+
- Webcam (for `collect_data.py`)
- Internet connection (first model download and YouTube pipeline)

Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

1. Verify your environment:

```bash
python _test_import.py
```

2. Collect manual webcam data:

```bash
python collect_data.py
```

3. Verify collected dataset:

```bash
python verify_data.py
```

4. Measure dataset readiness score:

```bash
python dataset_health_score.py
```

## YouTube Dataset Builder

Run full pipeline:

```bash
python youtube_dataset_builder.py
```

Common options:

```bash
python youtube_dataset_builder.py --limit 20
python youtube_dataset_builder.py --cats alfabe sayilar
python youtube_dataset_builder.py --verify
python youtube_dataset_builder.py --no-skip
```

Available categories:

- `alfabe`
- `sayilar`
- `icecekler`
- `aylar`
- `aile`
- `zamirler`
- `fiiller`

## Data Format

Expected sample format:

- File type: `.npy`
- Shape: `(30, 63)`
- Dtype: `float32`
- Meaning: 30 frames x 21 landmarks x 3 coordinates (x, y, z)

Directory example:

```text
data/
	MERHABA/
		0.npy
		1.npy
	EVET/
		0.npy
```

## Backlog Planning And Handoff

Generate class-based backlog and teammate assignment files:

```bash
python collection_backlog_planner.py --target 20 --top 120 --people 2 --daily-target-per-person 80
```

Outputs:

- `collection_backlog.csv`: Class counts and missing samples to target.
- `handoff_tasks.md`: Ready-to-use teammate assignment checklist.
- `NEXT_STEPS_DATA.md`: Operational handoff guide.

## Notes

- `hand_landmarker.task` is downloaded automatically into `models/` if not present.
- Root scripts resolve paths relative to the repository root (`data/`, `models/`).
- `youtube_dataset_builder.py` requires `yt-dlp` (included in `requirements.txt`).

## Troubleshooting

- If MediaPipe import fails:
	- Recreate venv and run `pip install -r requirements.txt`.
- If webcam cannot open:
	- Close other apps using camera and check `CAMERA_INDEX` in `collect_data.py`.
- If YouTube download fails:
	- Ensure internet access and retry.
	- Run `python -m yt_dlp --version` to verify installation.
- If verification reports shape errors:
	- Remove corrupted `.npy` files and recollect that class.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

## License

MIT. See `LICENSE` for details.

## Team

- Zehra Kelahmetoglu - Frontend
- Atakan Yilmaz - Data
- Zeynep Otegen - Optimization and Documentation
- Elifnur Gunay - Test and Maintenance
- Sevda Tuba Ehlibeyt - Backend
