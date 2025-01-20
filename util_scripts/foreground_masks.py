"""adapted from https://github.com/MischaD/chest-distillation"""

import os.path

import numpy as np
import torch
import torchvision
from einops import rearrange
from scipy.ndimage import binary_fill_holes, binary_closing
from sklearn.mixture import GaussianMixture

from util_scripts.utils_generic import resize_long_edge


class GMMMaskSuggestor:
    """
    A class for generating and refining segmentation masks using Gaussian Mixture Models (GMM).
    Useful for background/foreground separation and inpainting mask generation in images.
    """
    def __init__(self, opt):
        """
        Initializes the GMMMaskSuggestor with specified options.
        :param opt: Configuration options for model setup.
        """
        self.opt = opt
        self.gmm = GaussianMixture(n_components=2)

    def filter_orphan_pixel(self, img):
        """
        Applies a convolution to filter out isolated pixels in the image.
        :param img: Input image tensor.
        :return: Image tensor with orphan pixels filtered.
        """
        assert len(img.size()) == 2
        img = rearrange(img, "h w -> 1 1 h w").to(torch.float32)
        weights = torch.full((1, 1, 3, 3), 1 / 9)
        img[..., 1:-1, 1:-1] = torch.nn.functional.conv2d(img, weights.to(img.device),
                                                          bias=torch.zeros(1, device=img.device))
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        return img.squeeze()

    def get_gaussian_mixture_prediction(self, mask):
        """
        Fits a GMM to the mask and uses the means to create a binary thresholded image.
        :param mask: Input mask tensor.
        :return: Binary image tensor after applying GMM thresholding.
        """
        self.gmm.fit(rearrange(mask.to("cpu"), "h w -> (h w) 1"))
        threshold = np.mean(self.gmm.means_)
        binary_img = mask > threshold
        return binary_img

    def get_rectangular_inpainting_mask(self, segmentation_mask):
        """
        Get rectangular map of where to inpaint. Can overlap with object. In this case the object will not be inpainted
        and multiple inpainting spots are generate around the object.
        Input mask should contain False at all background and True at all Foreground pixels.
        Output will be masked that is True where inpainting is possible (part of background) and False where inpainting
        is not possible.

        :param segmentation_mask: binary mask where True is the foreground and False is the background we want to sample
               from
        :return:
        """
        binary_mask = segmentation_mask
        inpainting_mask = torch.zeros_like(binary_mask)
        if binary_mask.sum() >= ((binary_mask.size()[0] * binary_mask.size()[1]) - 1):
            # all foreground
            return inpainting_mask

        x, y = np.where(binary_mask is False)
        number_of_retries = 100  # after some attempts falls back to inpainting whole background
        while number_of_retries > 0:
            random_corner = np.random.randint(0, len(x))
            random_other_corner = np.random.randint(0, len(x))
            if random_corner == random_other_corner:
                continue

            # tl corner
            tl = (min(x[random_corner], x[random_other_corner]),
                  min(y[random_corner], y[random_other_corner]))

            # br corner
            br = (max(x[random_corner], x[random_other_corner]),
                  max(y[random_corner], y[random_other_corner]))

            width = br[0] - tl[0]
            height = br[1] - tl[1]
            area = width * height
            is_not_large_enough = (width <= 10 or height <= 10 or area < 16 ** 2)
            is_too_large = (width > 32 and height > 32) or area > 32 ** 2
            if (is_not_large_enough or is_too_large) and number_of_retries >= 0:
                number_of_retries -= 1
                continue

            box_location = torch.zeros_like(binary_mask)
            slice_ = (slice(tl[0], (br[0] + 1)), slice(tl[1], (br[1] + 1)))
            box_location[slice_] = True

            background_pixels = np.logical_and(np.logical_not(binary_mask), box_location)

            ratio = background_pixels.sum() / area
            if ratio < 2 / 3 and number_of_retries >= 0:
                # too many foreground pixels
                number_of_retries -= 1
                continue
            inpainting_mask = background_pixels
            break
        if number_of_retries == 0:
            inpainting_mask = np.logical_not(segmentation_mask)
        return inpainting_mask.to(bool)

    def __call__(self, sample, key="preliminary_mask"):
        """
        Processes the sample to filter orphan pixels and get GMM-based mask prediction.
        :param sample: Input sample containing image data.
        :param key: Key to access the preliminary mask in the sample.
        :return: Processed binary mask as a tensor.
        """
        if key is None:
            prelim_mask = sample
        else:
            prelim_mask = sample[key]
        prelim_mask = self.get_gaussian_mixture_prediction(prelim_mask.squeeze())
        orphan_filtered = self.filter_orphan_pixel(prelim_mask)
        return orphan_filtered.to(bool)

    def refined_mask_suggestion(self, sample):
        """
        Refines the mask suggestion by considering both the preliminary and inpainted images.
        :param sample: Dictionary containing the sample data.
        :return: Refined mask tensor.
        """
        if sample.get("preliminary_mask") is None:
            raise ValueError("Preliminary mask not part of sample dict - please call FOBADataset.add_preliminary_masks")
        if sample.get("inpainted_image") is None:
            raise ValueError("Inpainted Image not part of sample dict - please call FOBADataset.add_inpaintings")

        tmp_mask_path = os.path.join(self.opt.base_dir, "refined_mask_tmp", sample["rel_path"] + ".pt")
        if os.path.isfile(tmp_mask_path):
            return torch.load(tmp_mask_path)

        # get original image
        original_image = sample["img"]
        original_image = (original_image + 1) / 2
        y_resized = resize_long_edge(original_image, 512)  # resized s.t. long edge has lenght 512
        # y = torch.zeros((1, 3, 512, 512))
        # y[:, :, :y_resized.size()[-2], :y_resized.size()[-1]]

        # get preliminary mask
        resize_to_img_space = torchvision.transforms.Resize(512)
        prelim_mask = resize_to_img_space(self(sample, "preliminary_mask").unsqueeze(dim=0))

        # get inpainted image
        inpainted = sample["inpainted_image"]

        # get gmm diff mask
        diff = abs(y_resized - inpainted)
        diff = rearrange(diff, "1 c h w -> 1 h w c")
        diff_mask = self(diff.mean(dim=3), key=None).unsqueeze(dim=0)

        prelim_mask = prelim_mask[:, :diff_mask.size()[-2], :diff_mask.size()[-1]]
        refined_mask = prelim_mask * diff_mask
        refined_mask = refined_mask.unsqueeze(dim=0)
        os.makedirs(os.path.dirname(tmp_mask_path), exist_ok=True)
        torch.save(refined_mask, tmp_mask_path)
        return refined_mask

    def _compute_post_processed(self, refined_mask):
        """
        Post-processes the refined mask to fill holes and close small gaps.
        :param refined_mask: Refined mask tensor.
        :return: Post-processed mask tensor.
        """
        refined_mask = refined_mask.squeeze()
        refined_mask = torch.tensor(binary_fill_holes(binary_closing(refined_mask.cpu()))).to("cuda")
        refined_mask = rearrange(refined_mask, "h w -> 1 1 h w ")
        return refined_mask

    def postprocessd_refined_mask_suggestion(self, sample):
        """
        Finalizes the refined mask suggestion by applying post-processing.
        :param sample: Dictionary containing the sample data.
        :return: Post-processed refined mask tensor.
        """
        refined_mask = self.refined_mask_suggestion(sample)
        return self._compute_post_processed(refined_mask)
