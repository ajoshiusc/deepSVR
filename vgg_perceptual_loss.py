import torch
import torchvision

#Perceptual loss is detived from code here:
#https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, im_size=224):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.IM_SIZE = im_size

    def forward2D(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(
                self.IM_SIZE, self.IM_SIZE), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(
                self.IM_SIZE, self.IM_SIZE), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

    def forward3D(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        loss = 0
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1, 1)
            target = target.repeat(1, 3, 1, 1, 1)

        # loop over x dir
        x_size = input.shape[2]
        for i in range(x_size):
            loss += self.forward2D(input[:, :, i, :, :],
                                   target[:, :, i, :, :], feature_layers, style_layers)

        # loop over y dir
        y_size = input.shape[3]
        for i in range(y_size):
            loss += self.forward2D(input[:, :, :, i, :],
                                   target[:, :, :, i, :], feature_layers, style_layers)

        # loop over z dir
        z_size = input.shape[4]
        for i in range(z_size):
            loss += self.forward2D(input[:, :, :, :, i],
                                   target[:, :, :, :, i], feature_layers, style_layers)

        return loss


    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):

        if len(input.shape)==4:
            loss = self.forward2D(input, target, feature_layers=feature_layers, style_layers=style_layers)
        elif len(input.shape)==5:
            loss = self.forward3D(input, target, feature_layers=feature_layers, style_layers=style_layers)

        return loss

