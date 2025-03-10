import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_loader(image, img_size):
    image = Image.open(image)
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image = img_transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(torch.tensor([0.485, 0.456, 0.406]).to(device)).view(-1, 1, 1)
        self.std = torch.tensor(torch.tensor([0.229, 0.224, 0.225]).to(device)).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

normalization = Normalization().to(device)

VGG19_pretrained = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
VGG19_pretrained.to(device)

style_layers = ['conv_1', 'conv_2', 'conv_3']
def get_features(pretrained_model, image):
    layers = {
        '0': 'conv_1',
        '5': 'conv_2',
        '10': 'conv_3',
    }

    features = {}
    x = image
    x = normalization(x)
    for name, layer in pretrained_model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def get_dual_styles(style_features_1, style_features_2, style_layers):
    dual_style_features = {}
    for layer in style_layers:
        sf1 = style_features_1[layer]
        sf2 = style_features_2[layer]
        sf1_size = int(sf1.size()[1] / 2)
        dual_style_features[layer] = torch.concat((sf1[:, :sf1_size, :, :], sf2[:, sf1_size:, :, :]), dim=1)

    return dual_style_features

ContentLoss = nn.MSELoss()
StyleLoss = nn.MSELoss()

def style_tranfer_(model, optimizer, target_img,
                    content_features, style_features,
                    style_layers, content_weight, style_weight):
    """Performs one step of style transfer."""

    optimizer.zero_grad()

    target_features = get_features(model, target_img)

    content_loss = ContentLoss(content_features['conv_3'],
                                        target_features['conv_3'])

    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += StyleLoss(style_gram, target_gram)

    total_loss = content_loss*content_weight + style_loss*style_weight
    total_loss.backward(retain_graph=True)
    optimizer.step()
    return total_loss, content_loss, style_loss

# --- Streamlit App ---
st.title("Neural Style Transfer (Dual-Style)")
st.sidebar.header("Upload Images")
style_image1_file = st.sidebar.file_uploader("Choose Style Image 1", type=["jpg", "jpeg", "png"])
style_image2_file = st.sidebar.file_uploader("Choose Style Image 2", type=["jpg", "jpeg", "png"])
content_image_file = st.sidebar.file_uploader("Choose Content Image", type=["jpg", "jpeg", "png"])
img_size = st.sidebar.number_input("Image Size", min_value=100, max_value=512, value=224, step=1)
content_weight = st.sidebar.number_input("Content Weight", min_value=0.0, max_value=1e6, value=1.0, step=0.1)
style_weight = st.sidebar.number_input("Style Weight", min_value=0.0, max_value=1e6, value=1.0, step=0.1)
text = st.sidebar.text("Recommended Style Weight: 1e6")
steps = st.sidebar.number_input("Number of Steps", min_value=100, max_value=1000, value=300, step=100) # Changed to number_input


st.header("Uploaded Images")
cols = st.columns(3)
if style_image1_file:
    style_img1_pil = Image.open(style_image1_file)
    cols[0].image(style_img1_pil, caption="Style Image 1", use_container_width =True)
if style_image2_file:
    style_img2_pil = Image.open(style_image2_file)
    cols[1].image(style_img2_pil, caption="Style Image 2", use_container_width =True)
if content_image_file:
    content_img_pil = Image.open(content_image_file)
    cols[2].image(content_img_pil, caption="Content Image", use_container_width =True)


if st.button("Run Style Transfer"):
    if not style_image1_file or not style_image2_file or not content_image_file:
        st.error("Please upload all style and content images.")
    else:
                style_img1 = image_loader(style_image1_file, img_size)
                style_img2 = image_loader(style_image2_file, img_size)
                content_img = image_loader(content_image_file, img_size)

                content_features = get_features(VGG19_pretrained, content_img)
                style_features1 = get_features(VGG19_pretrained, style_img1)
                style_features2 = get_features(VGG19_pretrained, style_img2)
                final_style_features = get_dual_styles(style_features1, style_features2, style_layers)

                target_img = content_img.clone().requires_grad_(True).to(device)
                optimizer = optim.Adam([target_img], lr=0.02)

                for step in range(steps):
                    total_loss, content_loss, style_loss = style_tranfer_(VGG19_pretrained, optimizer, target_img,
                                                                            content_features, final_style_features,
                                                                            style_layers, content_weight, style_weight)
                    if step % 100 == 99:
                        st.write(f"Epoch [{step+1}/{steps}] Total loss: {total_loss.item():.6f} - Content loss: {content_loss.item():.6f} - Style loss: {style_loss.item():.6f}")

                with torch.no_grad():
                    target_img.clamp_(0, 1)
                output_image = target_img.cpu().clone()
                output_image = output_image.squeeze(0)
                output_image = transforms.ToPILImage()(output_image)

                st.header("Output Image")
                st.image(output_image, caption="Stylized Image", use_container_width=True)
                st.success("Style transfer complete!")
