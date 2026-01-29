from models.mmaf import build_model
from utils.arg_parser import get_argparser
from utils.carpk_arg_parser import carpk_get_argparser
import numpy as np
import random
import argparse
import os

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.ops import roi_align
from PIL import Image

class CARPKDataset(Dataset):
    def __init__(
            self, data_path, img_size, split='train'
    ):

        self.split = split
        self.data_path = data_path
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size))

        with open(os.path.join(
                self.data_path,
                'ImageSets',
                'train.txt' if split == 'train' else 'test.txt'
        )) as file:
            self.image_names = [line.strip() for line in file.readlines()]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(
            self.data_path,
            'Images',
            self.image_names[idx] + '.png'
        )).convert("RGB")

        w, h = img.size

        if self.split != 'train':
            img = T.Compose([
                T.ToTensor(),
                self.resize,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(img)


        with open(os.path.join(
            self.data_path,
            'Annotations',
            self.image_names[idx] + '.txt'
        )) as file:

            annotations = [list(map(int, line.strip().split())) for line in file.readlines()]

            bboxes = torch.tensor(annotations)[:, :-1]

        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        return img, bboxes

    def __len__(self):
        return len(self.image_names)



def extract_queries(model, x, bboxes):

    num_objects = bboxes.size(1) if not model.zero_shot else model.num_objects

    backbone_features = model.backbone(x)
 
    src = model.input_proj(backbone_features)
    src = model.msf(src)
   
    bs, c, h, w = src.size()
   
    pos_emb = model.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)

    src = src.flatten(2).permute(2, 0, 1)

    if model.num_encoder_layers > 0:

        image_features = model.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
    else:
        image_features = src


    f_e = image_features.permute(1, 2, 0).reshape(-1, model.emb_dim, h, w)

    if not model.zero_shot:

        box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 5).to(bboxes.device)

        box_w = bboxes[:, :, 2] - bboxes[:, :, 0]
        box_h = bboxes[:, :, 3] - bboxes[:, :, 1]
        perimeter = 2 * (box_h + box_w)
        area = box_h * box_w
        box_hw[:, :, 0] = box_w
        box_hw[:, :, 1] = box_h
        box_hw[:, :, 2] = box_h / (box_w + 1e-6)
        box_hw[:, :, 3] = torch.sqrt(box_w ** 2 + box_h ** 2)
        box_hw[:, :, 4] = perimeter ** 2 / (4 * torch.pi * area)


        shape_or_objectness = model.ffm.shape_or_objectness(box_hw).reshape(
            bs, -1, model.kernel_dim ** 2, model.emb_dim
        ).flatten(1, 2).transpose(0, 1)

    else:
        shape_or_objectness = model.ffm.shape_or_objectness.expand(
            bs, -1, -1, -1
        ).flatten(1, 2).transpose(0, 1)

    if not model.zero_shot:

        bboxes = torch.cat([
            torch.arange(bs, requires_grad=False).to(bboxes.device).repeat_interleave(num_objects).reshape(-1, 1),
            bboxes.flatten(0, 1),
        ], dim=1)

        appearance = roi_align(
            f_e,
            boxes=bboxes, output_size=model.kernel_dim,
            spatial_scale=1.0 / model.reduction, aligned=True
        ).permute(0, 2, 3, 1).reshape(
            bs, num_objects * model.kernel_dim ** 2, -1
        ).transpose(0, 1)

    else:
        appearance = None

    return shape_or_objectness, appearance



def predict(model, x, shape_or_objectness, appearance, num_objects):


    backbone_features = model.backbone(x)

    src = model.input_proj(backbone_features)
    src = model.msf(src)

    bs, c, h, w = src.size()

    pos_emb = model.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)

    src = src.flatten(2).permute(2, 0, 1)

    if model.num_encoder_layers > 0:

        image_features = model.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
    else:
        image_features = src

   
    f_e = image_features.permute(1, 2, 0).reshape(-1, model.emb_dim, h, w)


    query_pos_emb = model.ffm.pos_emb(
        bs, model.kernel_dim, model.kernel_dim, f_e.device
    ).flatten(2).permute(2, 0, 1).repeat(num_objects, 1, 1)


    if model.ffm.num_iterative_steps > 0:

        memory = f_e.flatten(2).permute(2, 0, 1)
 
        all_prototypes = model.ffm.multiattention_fused(
            shape_or_objectness, appearance, memory, pos_emb, query_pos_emb
        )
    else:
        if shape_or_objectness is not None and appearance is not None:
            all_prototypes = (shape_or_objectness + appearance).unsqueeze(0)
        else:
            all_prototypes = (
                shape_or_objectness if shape_or_objectness is not None else appearance
            ).unsqueeze(0)


    outputs = list()

    for i in range(all_prototypes.size(0)):
       
        prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
            bs, num_objects, model.kernel_dim, model.kernel_dim, -1
        ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

        response_maps = F.conv2d(
            torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
            prototypes,
            bias=None,
            padding=model.kernel_dim // 2,
            groups=prototypes.size(0)
        ).view(
            bs, num_objects, model.emb_dim, h, w
        ).max(dim=1)[0]

        if i == all_prototypes.size(0) - 1:
            predicted_dmaps = model.regression_head(response_maps)
            outputs.append(predicted_dmaps)


    return outputs[-1]


@torch.no_grad()
def eval_carpk(model, args, num_objects=12):
    train = CARPKDataset(
        args.data_path, img_size=512, split='train'
    )

    train_objects = list()
    for i in range(len(train)):  # len(train):989
       
        _, bboxes = train[i]
        train_objects.extend((i, j) for j in range(bboxes.size(0)))

    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    # pretrained
    torch.manual_seed(6)

 
    bbox_idx = torch.randperm(len(train_objects))[:num_objects]

    bbox_idx = [train_objects[t] for t in bbox_idx]

    shape, appearance = list(), list()

    for i, (img_idx, object_idx) in enumerate(bbox_idx):

        img, bboxes = train[img_idx]
        img, bboxes = img.to(device), bboxes.to(device)
      
        bboxes = bboxes[[object_idx], :].unsqueeze(0)
      
        sh, app = extract_queries(model, img.unsqueeze(0), bboxes)
      
        shape.append(sh[:args.kernel_dim ** 2])
     
        appearance.append(app[:args.kernel_dim ** 2])

 
    shape, appearance = torch.cat(shape, dim=0), torch.cat(appearance, dim=0)

    mae, mse = 0, 0

    test = CARPKDataset(
        args.data_path, 512, split='test'
    )

    loader = DataLoader(
        test,
        batch_size=1,
        shuffle=False,
    )

    for img, bboxes in loader:
        img = img.to(device)
        target = torch.tensor(bboxes.size(1)).to(device)
        output = predict(model, img, shape, appearance, num_objects)

        mae += (output.sum() - target).abs().item() / len(test)
        mse += (output.sum() - target).pow(2).item() / len(test)

    print("MAE: %.2f, RMSE: %.2f \n"%(mae, torch.sqrt(torch.tensor(mse)).item()))


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser('CARPK_TRAIN', parents=[carpk_get_argparser()])

    args = parser.parse_args()
    model = build_model(args)
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    eval_carpk(model, args, num_objects=12)
