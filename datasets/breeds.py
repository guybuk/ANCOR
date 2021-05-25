import os

from robustness.tools import folder
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from robustness.tools.helpers import get_label_mapping


class BREEDSFactory:
    def __init__(self, info_dir, data_dir):
        self.info_dir = info_dir
        self.data_dir = data_dir

    def get_breeds(self, ds_name, partition, mode='coarse', transforms=None, split=None):
        superclasses, subclass_split, label_map = self.get_classes(ds_name, split)
        partition = 'val' if partition == 'validation' else partition
        print(f"==> Preparing dataset {ds_name}, mode: {mode}, partition: {partition}..")
        if split is not None:
            # split can be  'good','bad' or None. if not None, 'subclass_split' will have 2 items, for 'train' and 'test'. otherwise, just 1
            index = 0 if partition == 'train' else 1
            return self.create_dataset(partition, mode, subclass_split[index], transforms)
        else:
            return self.create_dataset(partition, mode, subclass_split[0], transforms)

    def create_dataset(self, partition, mode, subclass_split, transforms):
        coarse_custom_label_mapping = get_label_mapping("custom_imagenet", subclass_split)
        fine_subclass_split = [[item] for sublist in subclass_split for item in sublist]
        fine_custom_label_mapping = get_label_mapping("custom_imagenet", fine_subclass_split)
        if mode == 'coarse':
            active_custom_label_mapping = coarse_custom_label_mapping
            active_subclass_split = subclass_split
        elif mode == 'fine':
            active_custom_label_mapping = fine_custom_label_mapping
            active_subclass_split = fine_subclass_split
        else:
            raise NotImplementedError

        dataset = folder.ImageFolder(root=os.path.join(self.data_dir, partition), transform=transforms,
                                     label_mapping=active_custom_label_mapping)
        coarse2fine = self.extract_c2f_from_dataset(dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition)
        setattr(dataset, 'num_classes', len(active_subclass_split))
        setattr(dataset, 'coarse2fine', coarse2fine)
        return dataset

    def extract_c2f_from_dataset(self, dataset,coarse_custom_label_mapping,fine_custom_label_mapping,partition):
        classes, original_classes_to_idx = dataset._find_classes(os.path.join(self.data_dir, partition))
        _,coarse_classes_to_idx = coarse_custom_label_mapping(classes, original_classes_to_idx)
        _, fine_classes_to_idx = fine_custom_label_mapping(classes, original_classes_to_idx)
        coarse2fine={}
        for k,v in coarse_classes_to_idx.items():
            if v in coarse2fine:
                coarse2fine[v].append(fine_classes_to_idx[k])
            else:
                coarse2fine[v] = [fine_classes_to_idx[k]]
        return coarse2fine

    def get_classes(self, ds_name, split=None):
        if ds_name == 'living17':
            return make_living17(self.info_dir, split)
        elif ds_name == 'entity30':
            return make_entity30(self.info_dir, split)
        elif ds_name == 'entity13':
            return make_entity13(self.info_dir, split)
        elif ds_name == 'nonliving26':
            return make_nonliving26(self.info_dir, split)
        else:
            raise NotImplementedError