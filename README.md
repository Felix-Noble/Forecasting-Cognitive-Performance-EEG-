Markdown

# APTIMA EEG Analysis Pipeline

## 1. Project Overview

This project is an EEG analysis pipeline designed for the APTIMA project. It processes raw EGI data, converts it to BIDS format, preprocesses it, creates epochs, and performs a Hurst exponent analysis using bootstrapping. The pipeline is driven by a central `config.toml` file and executed through a command-line interface in `main.py`, which allows for running specific pipeline steps individually.

## 2. File Structure

The project is organized into a `src` directory containing the core pipeline modules, with top-level files for configuration and execution.

~~~
├── main.py                 # Main entry point for running the analysis pipeline.
├── config.toml             # Configuration file for paths, settings, and pipeline parameters.
├── generate_config.py      # Generates a default config.toml file.
├── pyproject.toml          # Project metadata and build system configuration.
├── requirements.txt        # Lists the Python dependencies for the project.
└── src/
├── init.py
├── analysis/
│   ├── init.py
│   └── bootstrap.py        # Manages and runs the bootstrapping analysis.
├── data_preparation/
│   ├── init.py
│   ├── create_epochs.py    # Creates epochs from preprocessed data.
│   ├── egi_to_bids.py      # Converts raw EGI data to BIDS format.
│   └── preprocess.py       # Preprocesses BIDS data (filtering, ICA).
└── utils/
├── init.py
├── config_loader.py    # Helper to find project root and load config.toml.
└── make_loggers.py     # Helper to create console and file loggers.
~~~

## 3. Dependencies

The project requires several Python packages, including `mne`, `mne-bids`, `numpy`, `pandas`, `nolds`, and `toml`. All dependencies can be installed using:

```bash
pip install -r requirements.txt
'''
4. Configuration (config.toml)
The pipeline is configured via config.toml. The src/utils/config_loader.py module robustly finds and loads this file by locating the project's root directory (marked by pyproject.toml). It provides functions to access different sections of the configuration, ensuring that paths are correctly resolved relative to the project root.

If config.toml is not found, main.py will automatically generate a default version.

5. Executing the Pipeline with main.py
The main.py script serves as the primary entry point and acts as a dispatcher for running different stages of the analysis. It maps command-line arguments to the corresponding pipeline functions.

Operations
Initialization: Upon running, it ensures a config.toml file exists, creating one if necessary.

Argument Mapping: It uses a dictionary to map recognized string commands to their respective functions from the src modules.

Execution: It iterates through the provided command-line arguments and executes the corresponding functions in order.

Available Commands
You can run any of the following pipeline steps by passing their name as an argument to main.py:

egi_to_bids: Converts raw EGI data to BIDS format.

preprocess: Applies filtering and ICA to the BIDS data.

create_epochs or epoch: Segments the preprocessed data into epochs.

per level hurst bootstrap or hurst boot: Runs the final Hurst exponent bootstrap analysis.

help or h: Displays the list of available commands.