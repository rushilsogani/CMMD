import torch
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from mmd import MMD

class CLIP:
  
  def __init__(self):
    """Initialize CLIP model and processor with the correct variant"""
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
    self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")  # Fixed processor
  
  def image_and_text_feature(self):
    """Extracts CLIP features from text and images and computes MMD"""
    
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_inputs = tokenizer(["a parrot"], padding=True, return_tensors="pt").to(self.device)
    text_features = self.model.get_text_features(**text_inputs)

    #url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Pantheon_Rom_1_cropped.jpg/1920px-Pantheon_Rom_1_cropped.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)

    for i in range(2, 5):
      image = Image.open("./images/distorted" + str(i) + ".png")
      image2 = Image.open("./images/cleanParrot.png")

      image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
      image_inputs_2 = self.processor(images=image2, return_tensors="pt").to(self.device)
      image_features_1 = self.model.get_image_features(**image_inputs)
      image_features_2 = self.model.get_image_features(**image_inputs_2)

      # Normalizing the features seems to make the CMMD score very rigid (doesn't move much)
      # image_features_1 = image_features_1 / image_features_1.norm(p=2, dim=-1, keepdim=True) 
      # image_features_2 = image_features_2 / image_features_2.norm(p=2, dim=-1, keepdim=True)
      
      text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
      text_distribution = text_features.cpu().detach().numpy()

      image_distribution_1 = image_features_1.cpu().detach().numpy()
      image_distribution_2 = image_features_2.cpu().detach().numpy()

      mmd = MMD(X = image_distribution_1.T, Y = image_distribution_2.T, sigma = 0.5)
      comparison = mmd.compute_mmd()
      scale = 100
      print(f"The comparison between the text and image is: {scale * comparison}")

def main():
  clip = CLIP()
  clip.image_and_text_feature()  

if __name__ == "__main__":
  main()
