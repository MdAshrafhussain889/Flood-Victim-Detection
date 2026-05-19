# ============================================================
# explainability/gradcam.py
# Grad-CAM Explainability Module
# ============================================================

import cv2
import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor, output):
        self.model.zero_grad()
        output.backward(torch.ones_like(output))
        gradients  = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (256, 256))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
