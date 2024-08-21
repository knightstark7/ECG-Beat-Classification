from os.path import isdir
import subprocess
from config import PROJECT_DIR
import os

def bxb_script(bashlink, dir_path, dir_report, ext_ref, ext_ai, fileout, fileout_stan):
    """

    :param bashlink: 
    :param dir_path: 
    :param dir_report:
    :param ext_ref: 
    :param ext_ai: 
    :param fileout: 
    :param fileout_stan: 
    :return: 
    """
    subprocess.call(bashlink + ' ' +
                    dir_path + '/ ' +       # $1
                    dir_report + '/ ' +     # $2
                    ext_ref + ' ' +         # $3
                    ext_ai + ' ' +          # $4
                    fileout + ' ' +         # $5
                    fileout_stan + ' ',     # $6
                    shell=True)

def ec57_eval(output_ec57_directory,
              annotation_dir,
              beat_ext_db,
              event_ext_db,
              beat_ext_ai,
              event_ext_ai,
              fs_origin=360):
    """
    :param output_ec57_directory:
    :param annotation_dir:
    :param beat_ext_db:
    :param event_ext_db:
    :param beat_ext_ai:
    :param event_ext_ai:
    :param half_ext:
    :param fs_origin:
    :return:
    """
    dataset = 'mitdb'
    print(output_ec57_directory)
    print(dataset)
    if not isdir(output_ec57_directory):
        os.makedirs(output_ec57_directory)

    if beat_ext_ai is not None:
        bxb_script(PROJECT_DIR + 'script' + '/bxb-script.sh',
                   annotation_dir,
                   output_ec57_directory,
                   beat_ext_db,
                   beat_ext_ai,
                   dataset + '_qrs_report_line',
                   dataset + '_report_standard')
