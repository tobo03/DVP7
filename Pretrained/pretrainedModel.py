from torchvision import models, transforms
import torch.nn as nn

class pretrainedModel(nn.Module):
    def __init__(self):
        super(pretrainedModel, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)   # Imports the pretrained model.
        self.model = nn.Sequential(*list(self.model.children())[:-1])           # Removes the last layer from the model (The decision layer)

        for param in self.model.parameters():
            param.requires_grad = False

        self.scale = transforms.Resize((224, 224))  # Used to scale the image
        self.to_tensor = transforms.ToTensor()      # Used to transform the image to tensors

    def forward(self, x):
        
        x = self.scale(x)       # Scales from 28x28 --> 224x224
        x = self.to_tensor(x)   # Makes it to a tensor.
        x = x.unsqueeze(0)      # Adds a ditional dimension #? remove this if you don't give single images, but batches of images.
        x = self.model(x)       # Inputs the data into the altered pretrained model.
        x = x.flatten()         # Flattens the output to a single vector.
        
        return x

if __name__ == "__main__":
    model = pretrainedModel()

    from tensorflow import keras
    from PIL import Image
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    image = Image.fromarray(x_train[0]).convert('RGB')
    print(f"\n\n output --> {model(image)}\n\n")

    #images = [Image.fromarray(x_train[i]).convert('RGB') for i in range(10)]
    #print(f"\n\n output --> {model(image)}\n\n")
    #for image in images:
    #    print(f"\n\n output --> {model(image)}\n\n")