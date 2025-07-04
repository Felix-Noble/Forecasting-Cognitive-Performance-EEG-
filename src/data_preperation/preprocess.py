from src.utils.config_loader import get_paths, get_settings_prep, get_settings
from src.utils.make_loggers import make_logger
import mne
import os
from mne.preprocessing import ICA, create_eog_epochs
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mne_bids import (
    BIDSPath,
    find_matching_paths,
    write_raw_bids,
    read_raw_bids,
)
mne.set_log_level("CRITICAL")

class EventSort:
    def __init__(self):
        self.event_letter_codes = ["X", "F", 'G', 'S', "I"]

    def find_events(self, raw, shortest_event=1, stim_channel='STI 014'):
        events, event_id = None, None
        try:
            if shortest_event is not None:
                events = mne.find_events(raw, shortest_event=shortest_event)
            else:
                events = mne.find_events(raw)
            event_id = raw.event_id
            return events, event_id
        except ValueError:

            try:
                events = mne.find_events(raw, stim_channel=stim_channel)
                event_id = raw.event_id
            except ValueError:
                events, event_id = mne.events_from_annotations(raw)

        return events, event_id
    
    def master_event_id_MOT(self):
            new_event_id = {}  # creates new event_id dict and assigns the 'correct, no misses' event to id:30

            # contains all prefixes to event codes, X=Fixation, F=Flash,
            # G=Move_0 (GO), S=Move_1 (STOP), I=Space_bar (INPUT)
            
            event_letter_codes = self.event_letter_codes
            rst_state_evts_old = [('eyec', 'eyeo')]
            rst_state_evts = [('CLS0', 'CLS1'), ('CLS2', 'CLS3'), ('OPN0', 'OPN1'), ('OPN2', 'OPN3')]

            ms_vals = []
            # creates all event names/codes for each possible 'miss' value (NM, N missed out of  M targets) from 1/1 to 9/9
            for x in range(2, 10):
                for y in range(0, x):
                    miss_val = 'NM'
                    miss_val = miss_val.replace('N', str(y))
                    miss_val = miss_val.replace('M', str(x))
                    new_name = 'MS' + miss_val
                    ms_vals.append(new_name)

            # creates all event names/codes for each prefix in 'event_letter_codes',
            # from lvl 01 to 50, coding each for Correct (C) of Miss (M)
            for i, letter in enumerate(event_letter_codes):
                for x in range(1, 99):
                    trial = str(f'0{x}')
                    trial = trial[-2:]
                    for y in range(1, 99):
                        lvl = str(f'0{y}')
                        lvl = lvl[-2:]
                        evt_name = f'{letter}_{lvl}_{trial}_CR00'
                        new_event_id[evt_name] = len(new_event_id) + 1
                        for ms_val in ms_vals:
                            evt_name = f'{letter}_{lvl}_{trial}_{ms_val}'
                            new_event_id[evt_name] = len(new_event_id) + 1

            # for x in range(0, 10):
            #     new_event_id["ERR" + str(x)] = len(new_event_id) + 1

            return new_event_id

    def get_subject_event_id(self, events, event_id):
            subject_event_id = {}
            event_id_name = {v:k for k,v in event_id.items()}

            for event in events:
                if event_id_name[event[2]] not in subject_event_id.keys():
                    subject_event_id[event_id_name[event[2]]] = event[2]

            return subject_event_id

    def sort_events_MOT(self, events, event_id, sfreq=1000):
        """
        :info: Extracts information from existing events, creates new events list and event id's for BESA to read

        event name = 'ABCD' (4 letter code outputted by MOT task and read by BESA)
        event code = integer value, read by besa as 'trigger_no' in .evt file and by mne as the event_id

        :param events: '[[sample_1, change_value, event_id] ... [sample_n, change_value, event_id]]' numpy array
        containing each event, as read by mne
        :param events_id_code: reversed event_id dictionary returned by mne,
        allows for specification of events by their names, instead of their id's
        :return new_events: numpy array copying format of mne events. No events are copied
        :return new_event_id: dictionary of event_id's for all expected values of the new events.
        WARNING: not all events will be in this dict, create specific id dict for each event set for use in mne.
        """
        self.din_times = []
        self.current_sfreq = sfreq
        corr_miss = False
        CRMS_errors, DIN_errors = 0, 0
        trial_count = 0
        new_events = []
        self.master_event_id = self.master_event_id_MOT()
        event_letter_codes = self.event_letter_codes
        event_id_name = {v:k for k,v in event_id.items()}
        new_event_id = self.master_event_id
        new_event_id_name = {v:k for k,v in new_event_id.items()}

        # scrubs events list for information, collecting the location of fixation, flash, move1, move0,
        # and space_bar, writes new events that encode these events at their DIN1 signal (actual screen refresh time)
        # alongside lvl, corr/miss and if missed, n out of m targets

        index_dict = {"FX": None, "FLSH": None, "MVE0": None, "MVE1": None, 'SPCE': None}

        for i, event in enumerate(events):
            index_dict.update((k, None) for k in index_dict.keys())
            evt_id = event[2]
            evt_name = event_id_name[evt_id]
            for key in index_dict.keys():
                if evt_name.startswith(key):
                    index_dict[key] = i

            fx_index = index_dict['FX']
            if fx_index is not None:
                corr_miss = False
                trial_count += 1
                lvl = evt_name.replace('F', '')
                lvl = lvl.replace('X', '')
                lvl = f'0{lvl}'[-2:]
                for j, c_m_evt in enumerate(events[fx_index:fx_index + 100]):
                    curr_samp = c_m_evt[0]
                    fx_samp = events[fx_index][0]
                    t_diff = (curr_samp - fx_samp) / sfreq
                    if t_diff <= 17:
                        evt_id = c_m_evt[2]
                        c_m_name = event_id_name[evt_id]
                        if c_m_name.startswith('CRC'):
                            corr_miss = 'CR00'
                            break
                        elif c_m_name.startswith('MS'):
                            corr_miss = c_m_name
                            break
                    else:
                        break
                if not corr_miss:
                    CRMS_errors += 1

            index = 0
            if corr_miss is not False:
                for key, value in index_dict.items():
                    if value is not None:
                        for j, din_evt in enumerate(events[value:value + 100]):
                            curr_samp = din_evt[0]
                            evt_samp = events[value][0]
                            t_diff = (curr_samp - evt_samp) / sfreq
                            if t_diff <= 0.3:
                                evt_id = din_evt[2]
                                evt_name = event_id_name[evt_id]
                            else:
                                DIN_errors += 1
                                break
                            if evt_name.startswith('DIN'):
                                self.din_times.append(t_diff)
                                din_sample = np.float64(din_evt[0])
                                din_change = din_evt[1]

                                trial = f'0{trial_count}'
                                trial = trial[-2:]
                                new_evt_name = f'{event_letter_codes[index]}_{lvl}_{trial}_{corr_miss}'
                                new_evt_id = new_event_id[new_evt_name]
                                orders_of_mag = 1000 / self.current_sfreq
                                new_evt_sample = np.int64(din_sample * orders_of_mag)
                                new_evt = np.array([new_evt_sample, din_change, new_evt_id], dtype=np.int64)
                                new_events.append(new_evt)
                                break

                    index += 1

        
        new_events = np.vstack(new_events)
        
        subject_event_id = self.get_subject_event_id(new_events, new_event_id)
        return new_events, subject_event_id

def _apply_filter(raw, settings, settings_prep):
    """Applies notch and band-pass filters."""
    raw.load_data()
    raw.notch_filter(freqs=settings_prep["notch_filter"], 
                    picks=settings["channel_types"],
                    verbose=False)
    raw.filter(l_freq=settings_prep["low_filter"], 
                h_freq=settings_prep["high_filter"],
                method=settings_prep["filter_method"],
                verbose=False)
    raw.set_eeg_reference(ref_channels=settings_prep["reference"],
                        verbose=False)
    return None

def _run_ica(raw, settings, settings_prep):
    """Fits ICA and removes EOG components."""

    raw.set_channel_types({settings_prep["V_EOG_channel"]: 'eog',
                           settings_prep["H_EOG_anode"]: 'eog',
                           settings_prep["H_EOG_cathode"]: 'eog'})

    mne.set_bipolar_reference(raw,
                              anode=settings_prep["H_EOG_anode"],
                              cathode=settings_prep["H_EOG_cathode"],
                              ch_name='HEOG_bipolar',
                              copy=False, # Add to the current raw_with_eog_channels object
                              verbose=False)
    
    raw.set_channel_types({'HEOG_bipolar': 'eog'})

    ica = ICA(n_components=settings_prep['ica_n_components'], random_state=settings["random_state"])
    ica.fit(raw, picks=['eeg'])

    all_eog_indicies = []
    try: 
        eog_epochs_horizontal = create_eog_epochs(raw, ch_name='HEOG_bipolar',
                                                reject_by_annotation=False,
                                                thresh = settings_prep["EOG_thresh"])
        eog_indices_horizontal, eog_scores_horizontal = ica.find_bads_eog(eog_epochs_horizontal,
                                                                        ch_name='HEOG_bipolar')
        
        all_eog_indicies.extend(eog_indices_horizontal)
    
    except Exception as e:
        print("ERROR")
        raise e 

    try: 

        eog_epochs_vertical = create_eog_epochs(raw, ch_name=settings_prep["V_EOG_channel"],
                                            reject_by_annotation=False,
                                            thresh = settings_prep["EOG_thresh"])
        eog_indices_vertical, eog_scores_vertical = ica.find_bads_eog(eog_epochs_vertical,
                                                                ch_name=settings_prep["V_EOG_channel"])
        all_eog_indicies.extend(eog_indices_vertical)

    except Exception as e:
        print("ERROR")
        raise e 

    if len(all_eog_indicies) < 1:
        print("\t\tNO EOG INDICIES IDENTIFEID - HORIZONTAL AND VERTICAL FAILED")
        return None 
    
    
    ica.exclude = all_eog_indicies
    # bad_eog, scores = ica.find_bads_eog(raw, ch_name=settings_prep["EOG_CHANNEL"])
    ica.apply(raw)

    return None

def run_preprocessing_subject(bids_path: BIDSPath, settings: dict, settings_prep: dict, output_dir: Path, logger):
    """Filters, performs ICA EOG correction, and saves to derivatives directory"""
    logger.info(f"Preprocessing sub-{bids_path.subject}")
    raw = read_raw_bids(bids_path, 
                        verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    Event_processor = EventSort()
    if events.shape[0] < 1:
        events, event_id = Event_processor.find_events(raw)
    
    new_evt, new_evt_id = Event_processor.sort_events_MOT(events, event_id)
    del Event_processor, events, event_id
    print("Filtering")
    _apply_filter(  raw, 
                    settings, 
                    settings_prep)
    print("Fitting ICA")

    _run_ica(raw, 
                settings, 
                settings_prep)

    raw.set_annotations(None)

    out_path = BIDSPath(
        subject=bids_path.subject,#session=path.session,
        task=bids_path.task,
        run=bids_path.run,
        acquisition=bids_path.acquisition,
        processing=f"ICA{settings_prep['ica_n_components']}",  # Add a 'processing' label to distinguish
        suffix=bids_path.suffix,
        datatype=bids_path.datatype,
        root=output_dir,
        session = bids_path.session
        )

    write_raw_bids(
    raw=raw,
    bids_path=out_path,
    events=new_evt,
    event_id=new_evt_id,
    format="BrainVision",  # or your original format
    overwrite=True,
    allow_preload=True
    )
    logger.info(f"Subject {bids_path.subject} ICA{settings_prep['ica_n_components']} saved")
                            
def _preprocess_all_subjects():
    paths = get_paths()
    settings = get_settings()
    settings_prep = get_settings_prep()
    bids_root = paths["raw_bids_dir"]
    preprocessed_dir = Path(paths["preprocessed_dir"])

    bids_paths = find_matching_paths(bids_root, datatypes=settings["channel_types"], extensions=settings["extensions"])
    subjects_done = [x.subject for x in find_matching_paths(preprocessed_dir, datatypes=settings["channel_types"], extensions=settings["extensions"])]
    logger = make_logger(Path(__file__).stem)
    logger.info(f"Preprocessing subjects from {bids_root}\n Exporting to {preprocessed_dir}")
    logger.info(f"{len(subjects_done)} subjects finished, {len(bids_paths)-len(subjects_done)} remaining")

    for path in bids_paths:
        if path.subject in subjects_done:
            continue
        try:
            run_preprocessing_subject(path, settings, settings_prep, preprocessed_dir, logger)
        except Exception as e:
            logger.error(e)
       
if __name__ == "__main__":
    _preprocess_all_subjects()
    
