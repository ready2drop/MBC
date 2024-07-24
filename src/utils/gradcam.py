import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import models, transforms
from PIL import Image

# Load and preprocess the image
def get_img_array(img_path, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# Generate the Grad-CAM heatmap
def make_gradcam_heatmap(img_tensor, model, target_layer, class_idx=None):
    model.eval()

    # Hook the target layer to get the feature maps and gradients
    def forward_hook(module, input, output):
        global fmap
        fmap = output.detach()

    def backward_hook(module, grad_in, grad_out):
        global grad
        grad = grad_out[0].detach()

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output)
    
    # Zero gradients
    model.zero_grad()
    output[:, class_idx].backward()
    
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # Global average pooling of gradients
    grad_cam = torch.sum(weights * fmap, dim=1, keepdim=True)
    grad_cam = F.relu(grad_cam)
    grad_cam = F.interpolate(grad_cam, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='bilinear', align_corners=False)
    grad_cam = grad_cam - grad_cam.min()
    grad_cam = grad_cam / grad_cam.max()
    
    return grad_cam.squeeze().cpu().numpy()

# Superimpose the heatmap on the image
def save_and_display_gradcam(img_path, heatmap, cam_path="./grad_cam.jpg", alpha=0.4):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_heatmap = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_heatmap[heatmap]
    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((img.shape[1], img.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = Image.fromarray(np.uint8(superimposed_img))

    superimposed_img.save(cam_path)
    superimposed_img.show()

# Generate Grad-CAM for a given image and model
def Generate(path):
    # Load the pre-trained model
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[2].conv3  # Last convolutional layer
    
    # Preprocess the image
    img_tensor = get_img_array(path, size=(224, 224))

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_tensor, model, target_layer)

    # Save and display the Grad-CAM image
    save_and_display_gradcam(path, heatmap)

# Example usage
Generate("path/to/your/image.jpg")
