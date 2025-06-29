import os
import mne
import mne_bids
import mne_bids.read
import numpy as np
from src.utils.config_loader import get_paths, get_settings, get_settings_epoch
from pathlib import Path
import pandas as pd

from mne_bids import find_matching_paths

mne.set_log_level("CRITICAL")

def main(FolderName):
    errors = []
    paths = get_paths()
    settings = get_settings()
    settings_stage = get_settings_epoch()
    staging_root = Path(paths["staging_dir"]) / FolderName
    load_path = Path(paths["derivatives_dir"]) / "ICA"
    bids_paths = find_matching_paths(load_path, datatypes=settings["datatypes"], extensions=settings["extensions"])
    
    baseline_window = settings_stage["baseline_correct_window"]
    event_prefix = settings_stage["event_prefix"]

    for path in bids_paths:
        try: 
            print(f"Starting {path.subject}")
            if path.subject.endswith("99"):
                print("Skipping Subject")
                continue
            raw = mne_bids.read_raw_bids(path, verbose=False)
            raw.load_data()
            
            sfreq = raw.info["sfreq"]
            print(f"EEG loaded {path.subject}")

            subj_dir = Path(path).parent
            events_files = [x for x in list(subj_dir.rglob("*.tsv")) if "events" in str(x)]
            if len(events_files) != 1:
                raise ValueError(f"Events files identification error {len(events_files)} detected")
            
            event_df = pd.read_csv(events_files[0], sep="\t")
        
            # pull only Fixation events and those above level 3
            event_idx = [i for i in range(event_df.shape[0]) if event_df.loc[i,"trial_type"].startswith(event_prefix) and int(event_df.loc[i,"trial_type"].split("_")[1])>3]
            events_times = event_df.loc[event_idx,"onset"]
            IDs = event_df.loc[event_idx,"value"]
            events = np.vstack([np.array([x*sfreq,0,y], dtype=np.int32) for x,y in zip(events_times, IDs)])
            
            event_id = {k:v for k,v in zip(event_df.loc[event_idx, "trial_type"], event_df.loc[event_idx, "value"])}
            event_name_id = {v:k for k,v in event_id.items()}

            epochs = mne.Epochs(
                raw,
                events,
                event_id=None,
                tmin=baseline_window[0],
                tmax=10,
                baseline=baseline_window,
                preload=True,
                verbose=False
            )

            for i in range(len(epochs)):
                current_event_name = event_name_id[epochs.events[i,2]]
                level = current_event_name.split("_")[1]
                result = current_event_name.split("_")[-1]
                if result.startswith("CR"):
                    subdir = f"{level}C"
                else:
                    subdir = f"{level}M"

                output_filepath = staging_root / subdir 
                os.makedirs(output_filepath, exist_ok=True)
                epoch_data = epochs[i].get_data(copy=False).squeeze()
                
                np.save(output_filepath / f"sub-{path.subject}-{current_event_name}.npy", epoch_data)

        except Exception as e:  
            print(e)
            errors.append((path.subject,e))
    
if __name__ == "__main__":
    errors = main("PerLevel")
    from utils import save_error_log
    save_error_log(errors, "-select_data")