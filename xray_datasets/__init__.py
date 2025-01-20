"""
Init module for datasets.
"""

from .mimic import MimicCXRDataset, MimicCXRDatasetMSBBOX
from .xray14 import ChestXRay14BboxDataset


def get_dataset(opt, split=None):
    """
    Gets the correct dataset based on the provided config file.
    :param opt: A config file.
    :param split: Specifies the desired split (i.e. train, test, mscxr, p19).
    :return: The initialized and configured dataset.
    """
    datasets = {"chestxraymimic": MimicCXRDataset, "chestxraymimicbbox": MimicCXRDatasetMSBBOX,
                "chestxray14bbox": ChestXRay14BboxDataset}
    assert split is not None
    dataset_args = getattr(opt.datasets, f"{split}")
    getattr(opt, "dataset_args", dataset_args)
    dataset = datasets[dataset_args["dataset"]](dataset_args=dataset_args, opt=opt)
    return dataset
