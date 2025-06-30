#generate_config.py
import toml
from pathlib import Path
# from config import load_config
def config_generator():
    config_data = {
     'paths': {
                'derivatives_dir': 'derivatives', 
                'staging_dir': 'staging', 
                'raw_egi_dir': 'Path/to/data', 
                'raw_bids_dir': 'data/raw',
                'config_dir': 'config'}, 
               
    'settings': {
                'datatypes': ['eeg'], 
                'extensions': ['.vhdr']}, 

    'pipeline':{
        "preprocessing": {
                'notch_filter': [60], 
                'low_filter': 1, 
                'high_filter': 50, 
                'filter_method': 'fir', 
                'reference': 'average', 
                'ica_n_components': 15, 
                'EOG_CHANNEL': 'E126',
                't_minus_end': 3600}, 

    'epoch': {
        'event_prefix': 'F', 
        'baseline_correct_window': [-0.2, -0.1],
        'channels': list(range(125)),
        "time_max_s": 10}, 

    'bootstrap': {
            'n_samples': 1000, 
            'n_workers': 8, 
            'channels': list(range(125)), # include channels 1-125 for bootsrapping
            'time_range_ms':[0, 10100] # time range in ms where 0 is start of npy file, not start of epoch
            }, 
            }
            }
    
    with open(Path(config_data["paths"]["config_dir"])/ 'config.toml', 'w') as f:
        toml.dump(config_data, f)
