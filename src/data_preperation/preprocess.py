from src.utils.config_loader import get_paths, get_settings_prep, get_settings
from src.utils.make_loggers import make_logger
import mne
import os
from mne.preprocessing import ICA
from pathlib import Path

import matplotlib.pyplot as plt
from APEX_EEG_Processor import EEG_Processor
import numpy as np

from mne_bids import (
    BIDSPath,
    find_matching_paths,
    write_raw_bids,
    read_raw_bids,
)

def _apply_filter(raw, settings, settings_prep):
    """Applies notch and band-pass filters."""
    raw.load_data()
    raw.notch_filter(freqs=settings_prep["notch_filter"], 
                    picks=settings["datatypes"],
                    verbose=False)
    raw.filter(l_freq=settings_prep["low_filter"], 
                h_freq=settings_prep["high_filter"],
                method=settings_prep["filter_method"],
                verbose=False)
    raw.set_eeg_reference(ref_channels=settings_prep["reference"],
                        verbose=False)
    return None

def _run_ica(raw, settings_prep):
    """Fits ICA and removes EOG components."""
    ica = ICA(n_components=settings_prep['ica_n_components'])

    ica.fit(raw)
    bad_eog, scores = ica.find_bads_eog(raw, ch_name=settings_prep["EOG_CHANNEL"])
    ica.exclude = bad_eog
    ica.apply(raw)
    return None

def run_preprocessing_subject(bids_path: BIDSPath, settings: dict, settings_prep: dict, derivatives_dir: Path, logger):
    """Filters, performs ICA EOG correction, and saves to derivatives directory"""
    logger.info(f"Preprocessing sub-{bids_path.subject}")
    raw = read_raw_bids(bids_path, 
                        verbose=False)
    
    _apply_filter(  raw, 
                    settings, 
                    settings_prep)
    
    _run_ica(raw, 
                settings, 
                settings_prep)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    AP = EEG_Processor()
    if events.shape[0] < 1:
        events, event_id = AP.find_events(raw)
    
    new_evt, new_evt_id = AP.sort_events_MOT(events, event_id)
    new_evt = np.array(new_evt)
    del AP, events, event_id
    raw.set_annotations(None)

    ica_path = BIDSPath(
        subject=bids_path.subject,#session=path.session,
        task=bids_path.task,
        run=bids_path.run,
        acquisition=bids_path.acquisition,
        processing=f"ICA{settings_prep['ica_n_components']}",  # Add a 'processing' label to distinguish
        suffix=bids_path.suffix,
        datatype=bids_path.datatype,
        root=derivatives_dir / f"ICA-{settings_prep['ica_n_components']}"
        )

    write_raw_bids(
    raw=raw,
    bids_path=ica_path,
    events=new_evt,
    event_id=new_evt_id,
    format="BrainVision",  # or your original format
    overwrite=True,
    allow_preload=True
    )
    logger.info(f"Subject {bids_path.subject} ICA{settings_prep['ica_n_components']} saved")
                            
def main():
    paths = get_paths()
    settings = get_settings()
    settings_prep = get_settings_prep()
    bids_root = paths["raw_bids_dir"]
    bids_paths = find_matching_paths(bids_root, datatypes=settings["datatypes"], extensions=settings["extensions"])
    subjects_done = [x.subject for x in find_matching_paths(os.path.join(paths["derivatives_dir"],"ICA"),datatypes=settings["datatypes"], extensions=settings["extensions"])]
    logger = make_logger(Path(__file__).stem)

    logger.info(f"{len(subjects_done)} subjects finished, {len(bids_paths)-len(subjects_done)} remaining")

    for path in bids_paths:
        if path.subject in subjects_done:
            continue
        try:
            run_preprocessing_subject(path, settings, settings_prep, paths["derivatives_dir"], logger)

        except Exception as e:
            logger.error(f"sub-{path.subject} {e}")
    
if __name__ == "__main__":
    main()
    
