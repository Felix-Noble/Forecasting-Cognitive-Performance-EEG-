from config import get_paths, get_cut_settings 
import mne, os, glob, sys
from mne_bids import BIDSPath, write_raw_bids
from pathlib import Path
mne.set_log_level("WARNING")

def main():
    
    paths = get_paths()
    files = Path(paths["input_dir"]).rglob("*.mff")
    files_subjects = [(f,f.stem) for f in files]

    cut_settings = get_cut_settings()

    for (file, subject) in files_subjects:
        
        try:
            raw = mne.io.read_raw_egi(file, preload=False)
            last_time = (raw.n_times-1)/raw.info['sfreq'] # last timestamp = n. samples / sample frequency
            first_time = last_time - cut_settings["t_minus_end"]
            print(f"Processing {file}")
            print(f"First time: {first_time} last time {last_time}")
            if first_time > 0:
                raw = raw.crop(tmin=first_time, tmax=last_time)

            # mb = raw.get_data().nbytes / 1e6
            # print(mb)
            
            if str(subject).split("_")[-2:] in ["99"]:
                subject_id = str(subject).split("_")[0][:-2]
                session_id = "02"
            elif str(subject).split("_")[-2:] in ["88"]:
                subject_id = str(subject).split("_")[0][:-2]
                session_id = "01"
            else:
                subject_id = str(subject).split("_")[0]
                session_id = "01"
            task_name = "MOT"
            
            bids_path = BIDSPath(
                subject=subject_id,
                session=session_id,
                task=task_name,
                root=paths["output_dir"]
            )
            
            os.makedirs(Path(paths["output_dir"]) / f"sub-{subject_id}" / f"ses-{session_id}", exist_ok=True)
            
            write_raw_bids(raw, bids_path, overwrite=True)
            print(f"Subject: {subject_id} saved to {bids_path}")

        except Exception as e:
            print(e)

        
        
if __name__ == "__main__":
    main()