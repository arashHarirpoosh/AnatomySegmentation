import os
import libs
import shutil
from map_to_binary import class_map_5_parts

if __name__ == '__main__':
    groups = [x.split('_')[-1] for x in list(class_map_5_parts.keys())[:-1]]
    print(groups)
    root_path = 'Dataset'
    root_path_dest = 'DatasetCombined'
    all_folders = os.listdir(root_path)
    for s in all_folders:
        sub_path = f'{root_path}/{s}'
        all_files = os.listdir(sub_path)
        # print(s, all_files)
        for f in all_files:
            ct_file = f'{sub_path}/{f}/ct.nii.gz'
            segment_path = f'{sub_path}/{f}/segments'
            dest_path = f'{root_path_dest}/{s}/{f}'
            if os.path.isfile(ct_file):
                if not os.path.isdir(dest_path):
                    os.makedirs(dest_path)
                    os.makedirs(f'{dest_path}/segments')
                shutil.copy(src=ct_file, dst=f'{dest_path}/ct.nii.gz')
                libs.combine_masks_to_multilabel_file(masks_dir=segment_path,
                                                      multilabel_file=f'{dest_path}/segments/labels.nii.gz')
                for g in groups:
                    output_path = f'{dest_path}/segments/labels_task_{g}.nii.gz'
                    libs.combine_masks_to_multilabel_file_groups(masks_dir=segment_path,
                                                                 multilabel_file=output_path,
                                                                 class_type=g)
                print(s, f)

    print(all_folders)
    # libs.combine_masks_to_multilabel_file(masks_dir='', multilabel_file='')
