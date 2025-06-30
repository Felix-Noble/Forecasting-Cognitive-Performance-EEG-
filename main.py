#main.py
from src.data_preperation import egi_to_bids, preprocess, create_epochs
from src.analysis.bootstrap import run_per_level_hurst_bootsrap_analysis 

from src.utils.make_loggers import make_logger

import os
from pathlib import Path

func_dict = {"egi_to_bids" : egi_to_bids,
             "preprocess" : preprocess,
             "create_epochs": create_epochs,
             "epoch": create_epochs,
             "per level hurst bootstrap": run_per_level_hurst_bootsrap_analysis,
             "hurst boot": run_per_level_hurst_bootsrap_analysis}

def main(CLI_args):
    logger = make_logger(Path(__file__).stem)
    if not os.path.exists(Path(os.getcwd()) / "config" / "config.toml"):
        logger.info("No config found, initialising config from ./init/generate_config.py")
        from init.generate_config import config_generator
        config_generator()
    
    # check if 'help' present as argument
    for i in CLI_args:
        if i in ["help", "h"]:
            logger.info(f"Stored argument, function paris: \n{func_dict}")

    for arg in CLI_args:
        if arg in func_dict.keys():
            logger.info(f"Executing {arg}")
            func_dict[arg]()
        else:
            raise ValueError(f"{arg} not in recognised argument names, pass 'help' for more info")
    
if __name__ == "__main__":
    # TODO add CLI pass of arguments to main
    main(CLI_args=["hurst boot"])
