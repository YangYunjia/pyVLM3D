from .stdfun.run import StdRunArgs, standard_run
from .fileio.utils import dump_pretty_json

def standard_run_api(setting_file, verbose=False, debug=False, quiet=False):
    '''
    API to run VLM program

    ### params:

    - `setting_file`:    file path for settings
    
    '''

    args = StdRunArgs(setting_file, verbose, debug, quiet)
    results = standard_run(args)
    return results