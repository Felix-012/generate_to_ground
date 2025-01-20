"""addapted from https://github.com/MischaD/chest-distillation"""

import argparse
import json
import os

import torch
import torchxrayvision as xrv

from evaluation.inception import InceptionV3
from evaluation.xrv_fid import calculate_fid_given_paths
from log import logger

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def main(args):
    """
    Calculates the FID and FID_xrv scores between two datasets.
    :param args: Dictionary with arguments from get_args().
    :return:
    """
    device = torch.device('cuda')
    num_workers = args.num_workers

    results = {}
    dims = 0
    model = None
    for fid_model in ["inception", "xrv"]:
        if fid_model == "xrv":
            dims = 1024
            model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
        elif fid_model == "inception":
            dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(device)

        fid_value = calculate_fid_given_paths([args.path_src, args.path_tgt],
                                              args.batch_size,
                                              device,
                                              fid_model,
                                              model=model,
                                              dims=dims,
                                              num_workers=num_workers)
        logger.info(f"FID of the following paths: {args.path_src} -- {args.path_tgt}")
        logger.info(f'{fid_model} FID: {fid_value} --> ${fid_value: .1f}$')
        results[fid_model] = fid_value

    if hasattr(args, "result_dir") and args.result_dir is not None:
        with open(os.path.join(args.result_dir, "fid_results.json"), "w", encoding="utf-8") as file:
            results_file = {"dataset_src": args.path_src, "dataset_tgt": args.path_tgt}
            for fid_model, fid_value in results.items():
                results_file[fid_model] = {"FID": fid_value,
                                           "as_string": f"{fid_value: .1f}"
                                           }
            json.dump(results_file, file)


def get_args():
    """
    Get arguments for calc_fid.
    :return: Dictionary containing the passed arguments.
    """
    parser = argparse.ArgumentParser(description="Compute FID of dataset")
    parser.add_argument("path_src", type=str, help="Path to first dataset")
    parser.add_argument("path_tgt", type=str, help="Path to second dataset")
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of processes to use for data loading.')
    parser.add_argument("--result_dir", type=str, default=None, help="dir to save results in.")
    return parser.parse_args()


if __name__ == '__main__':
    arguments = get_args()
    main(arguments)
