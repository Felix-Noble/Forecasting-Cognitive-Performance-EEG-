#generate_config.py
import toml
from config import load_config

if __name__ == "__main__":
    config_data = {
    '__init__': {'init': True}, 
     
     'paths': {
                'root_dir': 'E:/Dropbox/4. Codebase/APTIMA/data', 
                'derivatives_dir': 'E:/Dropbox/4. Codebase/APTIMA/derivatives', 
                'staging_dir': 'E:/Dropbox/4. Codebase/APTIMA/staging', 
                'input_dir': 'D:/data/APTIMA dataset', 
                'output_dir': 'E:/Dropbox/4. Codebase/APTIMA/data'}, 
               
    'settings': {
                'datatypes': ['eeg'], 
                'extensions': ['.vhdr']}, 

    'settings_prep': {
                'notch_filter': [60], 
                'low_filter': 1, 
                'high_filter': 50, 
                'filter_method': 'fir', 
                'reference': 'average', 
                'ica_n_components': 15, 
                'EOG_CHANNEL': 'E126'}, 

    'settings_stage': {
        'event_prefix': 'F', 
        'baseline_correct_window': [-0.2, -0.1]}, 

    'settings_bootstrap': {
            'n_samples': 8, 
            'n_workers': 7, 
            "batch_size":7,
            'channels': list(range(125)), # include channels 1-125 for bootsrapping
            'time_range':[0, 10100] # time range in ms where 0 is start of npy file, not start of epoch
            }, 
            
    'cut_settings': {
        't_minus_end': 3600}
            
            }
    
    with open('config.toml', 'w') as f:
        toml.dump(config_data, f)
