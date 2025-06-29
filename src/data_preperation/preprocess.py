from config import get_paths, get_settings_prep, get_settings
import mne, os, glob, sys, glob
from mne.preprocessing import ICA
from pathlib import Path
# from utils import sort_events_MOT
import matplotlib.pyplot as plt
from APEX_EEG_Processor import EEG_Processor
import numpy as np

from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    write_raw_bids,
    read_raw_bids,
)

def filter(raw, settings, settings_prep):
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

def ica_process(raw, settings, settings_prep):
    ica = ICA(n_components=settings_prep['ica_n_components'])

    ica.fit(raw)
    bad_eog, scores = ica.find_bads_eog(raw, ch_name=settings_prep["EOG_CHANNEL"])
    ica.exclude = bad_eog
    ica.apply(raw)
    return None

def main():
    paths = get_paths()
    settings = get_settings()
    settings_prep = get_settings_prep()
    bids_root = paths["root_dir"]
    bids_paths = find_matching_paths(bids_root, datatypes=settings["datatypes"], extensions=settings["extensions"])
    subjects_done = [x.subject for x in find_matching_paths(os.path.join(paths["derivatives_dir"],"ICA"),datatypes=settings["datatypes"], extensions=settings["extensions"])]
    print(f"{len(subjects_done)} subjects finished, {len(bids_paths)-len(subjects_done)} remaining")

    errs=[]
    for path in bids_paths:
        if path.subject in subjects_done:
            continue
        print(f"Starting: {path.subject}")
        
        try:
            raw = read_raw_bids(path, 
                                verbose=False)
            filter(raw, settings, settings_prep)

            filtered_path = BIDSPath(
                subject=path.subject,#session=path.session,
                task=path.task,
                run=path.run,
                acquisition=path.acquisition,
                processing="FilteredAverageReference",  # Add a 'processing' label to distinguish
                suffix=path.suffix,
                datatype=path.datatype,
                root=os.path.join(paths["derivatives_dir"],"FilteredAverageReference"),
                )
            
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            AP = EEG_Processor()
            if events.shape[0] < 1:
                events, event_id = AP.find_events(raw)
            
            new_evt, new_evt_id = AP.sort_events_MOT(events, event_id)
            new_evt = np.array(new_evt)
            del AP, events, event_id
            raw.set_annotations(None)

            write_raw_bids(
            raw=raw,
            bids_path=filtered_path,
            events=new_evt,
            event_id=new_evt_id,
            format="BrainVision",  # or your original format
            overwrite=True,
            allow_preload=True
            )
            del filtered_path
            print(f"Subject {path.subject} filtered and saved")
            ica_process(raw, settings, settings_prep)

            ica_path = BIDSPath(
                subject=path.subject,#session=path.session,
                task=path.task,
                run=path.run,
                acquisition=path.acquisition,
                processing=f"ICA{settings_prep['ica_n_components']}",  # Add a 'processing' label to distinguish
                suffix=path.suffix,
                datatype=path.datatype,
                root=os.path.join(paths["derivatives_dir"], "ICA")
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
            del ica_path
            print(f"Subject {path.subject} ICA{settings_prep['ica_n_components']} saved")

        except Exception as e:
            errs.append((path.subject, str(e)))

    return errs

if __name__ == "__main__":
    errs = main()
    print(errs)
    from utils import save_error_log
    save_error_log(errs,"-preprocess")
