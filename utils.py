import os
import torch
import tempfile
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError
from pytorch_fid import fid_score

def calculate_metrics(real_images, generated_images, device):
    metrics = {}
    real_images = real_images.clamp(0, 1)
    generated_images = generated_images.clamp(0, 1)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    metrics['ssim'] = ssim(generated_images, real_images).item()
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    metrics['psnr'] = psnr(generated_images, real_images).item()
    mse = MeanSquaredError().to(device)
    metrics['mse'] = mse(generated_images, real_images).item()
    return metrics

def get_depth_map(images, processor, estimator, device, resolution):
    with torch.no_grad():
        depth_inputs = processor(images=images, return_tensors="pt").to(device)
        depth_outputs = estimator(**depth_inputs)
        predicted_depth = depth_outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(resolution, resolution),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(prediction, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(prediction, dim=[1, 2, 3], keepdim=True)
        prediction = (prediction - depth_min) / (depth_max - depth_min + 1e-8)
        control_depth = prediction.repeat(1, 3, 1, 1)
    return control_depth