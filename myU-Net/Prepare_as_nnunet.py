from pathlib import Path
from shutil import copy
import os
import multiprocessing as mp
from functools import partial
import json


def get_paths(path):
    """
    From a Path object (pathlib), the function return all the directories in this path

    input:
        - path: Path object (from pathlib)
    returns:
        - paths: all the paths from the directories inside path, sorted alphabetically
    """
    paths = []
    for i in path.iterdir():
        if i.is_dir():
            paths.append(i)
    return sorted(paths)

def get_files(path):
    """
    From a Path object (pathlib), the function return all the files in this path

    input:
        - path: Path object (from pathlib)
    returns:
        - files: all the files inside path, sorted alphabetically
    """
    files = []
    for i in path.iterdir():
        if i.is_file():
            files.append(i)
    return sorted(files)

path_data = Path()/'./data' # where the data are stored

paths = get_paths(path_data) # it will get all the folders "case_XXX"

new_path = Path('./nnUNet') # where you would like to write the files for nn-Unet

def copy_training_files(new_path,path):
    """
    From the Path objects (pathlib), the function copies the training data from path to new_path.
    Note that the data will be copied following the structure required by nn-Unet

    input:
        - path: Path object (from pathlib) where the KiTS19 is stored
        - new_path: where the data will be copied
    returns:
        - void, the function only copy the files
    """
    print('Copying: ',path.name)
    case_number = int(path.name.split('_')[1])
    copy(path/'imaging.nii.gz',new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'imagesTr'/'CtMO20_{:03d}_{:04d}.nii.gz'.format(case_number,0))
    copy(path /'segmentation.nii.gz',new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'labelsTr'/'CtMO20_{:03d}.nii.gz'.format(case_number))

def copy_test_files(new_path,path):
    """
    From the Path objects (pathlib), the function copies the test data from path to new_path.
    Note that the data will be copied following the structure required by nn-Unet

    input:
        - path: Path object (from pathlib) where the KiTS19 is stored
        - new_path: where the data will be copied
    returns:
        - void, the function only copy the files
    """
    print('Copying: ',path.name)
    case_number = int(path.name.split('_')[1])
    copy(path/'imaging.nii.gz',new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'imagesTs'/'CtMO20_{:03d}_{:04d}.nii.gz'.format(case_number,0))
    copy(path /'segmentation.nii.gz',new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'labelsTs'/'CtMO20_{:03d}.nii.gz'.format(case_number))

copy_fn_training = partial(copy_training_files,new_path) # to be able to run in parallel
copy_fn_test = partial(copy_test_files,new_path)

if __name__ == '__main__':
    #create directories
    os.makedirs(new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'imagesTr',exist_ok=True)
    os.makedirs(new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'labelsTr',exist_ok=True)
    os.makedirs(new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'imagesTs',exist_ok=True)
    os.makedirs(new_path / 'nnUNet_raw_data_base' / 'nnUNet_raw_data' / 'Task204_CtMO20' / 'labelsTs', exist_ok=True)
    #copy data
    with mp.Pool(4) as pool:
        pool.map(copy_fn_training,[i for i in paths[0:112]])
    with mp.Pool(4) as pool:
        pool.map(copy_fn_test,[i for i in paths[112:140]])
    #create the json file
    files_training = get_files(new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'imagesTr')
    files_label = get_files(new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'labelsTr')
    files_test = get_files(new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'imagesTs')
    json_dict = {}
    json_dict['name'] = "CtMO20"
    json_dict['description'] = "Segmentazioni di Fegato, Polmoni ed Ossa in immagini CT"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "CT-ORG data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Liver",
        "2": "Lungs",
        "3": "Bones"
    }
    json_dict['numTraining'] = len(paths[0:112])
    json_dict['numTest'] = len(paths[112:140])

    json_dict['training'] = [{   'image':str(new_path/ 'nnUNet_raw_data_base' / 'nnUNet_raw_data' / 'Task204_CtMO20' / 'imagesTr' / i.name),
                                 'label':str(new_path/ 'nnUNet_raw_data_base' / 'nnUNet_raw_data' / 'Task204_CtMO20' / 'labelsTr' / i.name)}
                             for i in files_label]

    json_dict['test'] = [str(   new_path/ 'nnUNet_raw_data_base' / 'nnUNet_raw_data' / 'Task204_CtMO20' / 'imagesTs' / (i.name[:-12]+'.nii.gz'))
                         for i in files_test]

    #save json file
    with open(new_path/'nnUNet_raw_data_base'/'nnUNet_raw_data'/'Task204_CtMO20'/'dataset.json', 'w') as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)

    os.makedirs(new_path/'nnUNet_preprocessed',exist_ok=True)
    os.makedirs(new_path/'RESULTS_FOLDER',exist_ok=True)
    #export paths (sometimes it does not work and we have to do it manually on the console)
    export_nnUNet_raw_data_base = 'export nnUNet_raw_data_base='+'"'+str(new_path/'nnUNet_raw_data_base')+'"'
    export_nnUNet_preprocessed = 'export nnUNet_preprocessed='+'"'+str(new_path/'nnUNet_preprocessed')+'"'
    export_RESULTS_FOLDER = 'export RESULTS_FOLDER='+'"'+str(new_path/'RESULTS_FOLDER')+'"'
    os.system(export_nnUNet_raw_data_base)
    os.system(export_nnUNet_preprocessed)
    os.system(export_RESULTS_FOLDER)


