import sys, string, random
from datetime import datetime
sys.path.append('../')
from omegaconf import OmegaConf
# import wandb
import torch
from monai.networks.nets import UNet

from data_utils import MNMv2DataModule
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.distributions import Categorical




# def __init__(self):
#     #TODO: load config

#     #init datamodule
#     mnmv2_config   = OmegaConf.load('configs/mnmv2.yaml')

#     data_dir = "../../lennartz/data/MNM/"
#     datamodule = MNMv2DataModule(
#         data_dir=data_dir,
#         vendor_assignment=mnmv2_config.vendor_assignment,
#         batch_size=mnmv2_config.batch_size,
#         binary_target=mnmv2_config.binary_target,
#         non_empty_target=mnmv2_config.non_empty_target,
#     )

#     datamodule.setup('test')
#     test_loader = datamodule.test_dataloader()



#     # load model using checkpoint instead of config file
#     checkpoint_path = 'checkpoints/mnmv2-11-52_29-10-2024.ckpt'


#     checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
#     model_state_dict = checkpoint['state_dict']
#     model_state_dict = {k.replace('model.model.', 'model.'): v for k, v in model_state_dict.items() if k.startswith('model.')}
#     model_config = checkpoint['hyper_parameters']['cfgs']



#     model = UNet(
#         spatial_dims=model_config['unet']['spatial_dims'],
#         in_channels=model_config['unet']['in_channels'],
#         out_channels=model_config['unet']['out_channels'],
#         channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
#         strides=[2] * (model_config['unet']['depth'] - 1),
#         num_res_units=4
#     )

#     model.load_state_dict(model_state_dict)

#     return model, test_loader


mnmv2_config   = OmegaConf.load('configs/mnmv2.yaml')

data_dir = "../../lennartz/data/MNM/"
datamodule = MNMv2DataModule(
    data_dir=data_dir,
    vendor_assignment=mnmv2_config.vendor_assignment,
    batch_size=mnmv2_config.batch_size,
    binary_target=mnmv2_config.binary_target,
    non_empty_target=mnmv2_config.non_empty_target,
)

datamodule.setup('test')
test_loader = datamodule.test_dataloader()

print("Data loaded")

def get_data_loader():
    #init datamodule
    mnmv2_config   = OmegaConf.load('configs/mnmv2.yaml')

    data_dir = "../../lennartz/data/MNM/"
    datamodule = MNMv2DataModule(
        data_dir=data_dir,
        vendor_assignment=mnmv2_config.vendor_assignment,
        batch_size=mnmv2_config.batch_size,
        binary_target=mnmv2_config.binary_target,
        non_empty_target=mnmv2_config.non_empty_target,
    )

    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    return test_loader


def load_model():
    # load model using checkpoint instead of config file
    checkpoint_path = 'checkpoints/mnmv2-11-52_29-10-2024.ckpt'


    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint['state_dict']
    model_state_dict = {k.replace('model.model.', 'model.'): v for k, v in model_state_dict.items() if k.startswith('model.')}
    model_config = checkpoint['hyper_parameters']['cfgs']



    model = UNet(
        spatial_dims=model_config['unet']['spatial_dims'],
        in_channels=model_config['unet']['in_channels'],
        out_channels=model_config['unet']['out_channels'],
        channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
        strides=[2] * (model_config['unet']['depth'] - 1),
        num_res_units=4
    )

    model.load_state_dict(model_state_dict)

    return model


def get_image_batch(batch_number):
    test_loader = get_data_loader()
    batch = test_loader[batch_number]

    return batch['input'].tolist()


def get_image(batch_number, img_number):
    # img = test_loader.dataset[batch_number]['input'][img_number]
    
    for i, batch in enumerate(test_loader):
        if i == batch_number:
            break

    img = batch['input'][img_number]

    return img.tolist()


def compute_entropy_dataframe(model, data_loader=test_loader):

    model = load_model()
    model = model.cuda() #move model to gpu
    model.eval() #set to eval mode. disables dropout layers too.
    
    statistics = []

    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            img_batch = batch['input'].cuda()
            
            #get predictions of all images in batch in parallel
            pred = model(img_batch)
            prob = softmax(pred)

            #Compute statistics for the predictictions
            error = (prob.argmax(dim=1)!=batch['target'].cuda().squeeze()).float()
            avg_error = error.mean(dim=(1,2)).tolist()
            entropy = Categorical(probs=prob.moveaxis(1, 3)).entropy()
            avg_entropies = entropy.mean(dim=(1, 2)).tolist()
            # Compute other uncertainty measures here
            #TODO


            #Generate index columns specifying batch and image_no
            batch_indices = torch.full((img_batch.shape[0],), i, dtype=torch.int)
            image_indices = torch.arange(img_batch.shape[0])
            
            #TODO add the new measures to the zip (Don't forget to give the column a name as well when constructing the dataframe)
            statistics.extend(
                zip(batch_indices.tolist(), image_indices.tolist(), avg_error, avg_entropies)
            )
            

    
    df = pd.DataFrame(statistics, columns=['batch_index', 'image_index', 'avg_error', 'avg_entropy'])
    df = df.sort_values(by='avg_error', ascending=False)
    return df