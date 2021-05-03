import torchvision
from torch import nn, optim

if __name__ == "__main__":

    # In finetuning, we freeze most of the model and typically only modify the classifier layers to make
    # predictions on new labels. Let’s walk through a small example to demonstrate this. As before,
    # we load a pretrained resnet18 model, and freeze all the parameters.

    model = torchvision.models.resnet18(pretrained=True)

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    print(model)

    # Let’s say we want to finetune the model on a new dataset with 10 labels. In resnet, the classifier is
    # the last linear layer model.fc. We can simply replace it with a new linear layer (unfrozen by default)
    # that acts as our classifier.

    model.fc = nn.Linear(512, 10)

    # Now all parameters in the model, except the parameters of model.fc, are frozen. The only parameters that
    # compute gradients are the weights and bias of model.fc.

    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)