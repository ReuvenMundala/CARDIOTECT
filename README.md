# Cardiotect Calcium Scoring

Automated coronary calcium scoring using deep learning with external validation.
Reproduces the method from npj Digital Medicine 2021.

## Quick Start (Windows)

Simply double-click `start_cardiotect_web.bat` in the root folder. It will:
1. Automatically run first-time setup if needed (creates virtual environment, installs dependencies)
2. Launch the main web interface in your default browser

That's it!

## What's Included

### Main Application
- **Web GUI** (`extras/web_gui/`): Browser-based interface for calcium scoring
- **Core AI** (`extras/cardiotect_cac/`): Deep learning model and scoring algorithms

### Optional Components
- **Desktop GUIs** (`extras/gui_v2/`, `extras/gui_v3/`): Alternative PyQt-based interfaces
- **CLI Tools**: Command-line training and inference capabilities

## Dataset Configuration

The dataset is **not included** in this repository. After first launch:

1. Edit the `.env` file in the project root
2. Set `CARDIOTECT_DATASET_ROOT` to your dataset path
3. Ensure your dataset follows this structure:
```
dataset/
└── cocacoronarycalciumandchestcts-2/
    └── Gated_release_final/
        ├── patient/
        │   └── {patient_id}/
        │       └── Pro_Gated_CS_3.0_I30f_3_70%/
        └── calcium_xml/
```

## Model Checkpoints
Pre-trained model checkpoints are included in `outputs/checkpoints/`. The web GUI will automatically load the best checkpoint. You can also train your own model using CLI: `python -m cardiotect_cac.train_cli --help`

## Project Structure
```
Cardiotect/
├── README.md                   # This file
├── start_cardiotect_web.bat    # Main launcher (run this!)
├── .env.example                # Configuration template
├── outputs/
│   └── checkpoints/            # Pre-trained model checkpoints
└── extras/                     # All application files
    ├── cardiotect_cac/         # Core AI module
    ├── web_gui/                # Web application
    ├── gui_v2/                 # Desktop GUI V2
    ├── gui_v3/                 # Desktop GUI V3
    ├── setup.bat               # First-time setup (auto-run by launcher)
    └── requirements.txt        # Python dependencies
```

## System Requirements
- Windows 10/11
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

## License
This project is for research purposes.
