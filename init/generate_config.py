#generate_config.py
import toml
from pathlib import Path
# from config import load_config
def config_generator():
    config_data = {
     'paths': {
                'derivatives_dir': 'derivatives', 
                'preprocessed_dir' : 'data/preprocessed',
                'staging_dir': 'staging', 
                'raw_egi_dir': 'Path/to/data', 
                'raw_bids_dir': 'data/raw',
                'config_dir': 'config'}, 
               
    'settings': {
                'channel_types': ['eeg'], 
                'extensions': ['.vhdr'],
                'random_state': 2025}, 

    'pipeline':{
        "preprocessing": {
                'notch_filter': [60], 
                'low_filter': 1, 
                'high_filter': 50, 
                'filter_method': 'fir', 
                'reference': 'average', 
                'ica_n_components': 120, 
                'V_EOG_channel': 'E17',
                'H_EOG_anode' : 'E126',
                'H_EOG_cathode': 'E127',
                't_minus_end': 3600}, 

    'epoch': {
        'event_prefix': 'X', 
        'baseline_correct_window': [0.05, 0.15],
        'channels': [x for x in range(125) if x not in [125, 126, 17]],
        "time_max_s": 10}, 

    'bootstrap': {
            'n_samples': 30, 
            'n_workers': 2, 
            'channels': [x for x in range(125) if x not in [125, 126, 17]], # include channels 1-125 for bootsrapping
            'time_range_ms':[0, 10100] # time range in ms where 0 is start of npy file, not start of epoch
            }, 
            }
            }
    
    with open(Path(config_data["paths"]["config_dir"])/ 'config.toml', 'w') as f:
        toml.dump(config_data, f)

if __name__ == "__main__":
    config_generator()