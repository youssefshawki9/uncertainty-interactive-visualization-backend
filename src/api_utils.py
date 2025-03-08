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

from threading import Thread
from collections import defaultdict

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


#Define the standard datamodule
data_dir = "../../lennartz/data/MNM/"
mnmv2_config   = OmegaConf.load('configs/mnmv2.yaml')
datamodule = MNMv2DataModule(
    data_dir=data_dir,
    vendor_assignment=mnmv2_config.vendor_assignment,
    batch_size=mnmv2_config.batch_size,
    binary_target=mnmv2_config.binary_target,
    non_empty_target=mnmv2_config.non_empty_target,
)

dataloader = None
def get_data_loader(datamodule=datamodule):
    global dataloader

    if dataloader: return dataloader

    datamodule.setup('test')
    dataloader = datamodule.test_dataloader()
    return dataloader


def load_model():
    # load model using checkpoint instead of config file
    checkpoint_path = 'checkpoints/mnmv2-11-52_29-10-2024.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)

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

stats_dataframe = None #Used by compute_stats_dataframe() as a global. Do not modify outside this scope!
status_stats_dataframe = "Not Generated"
def get_stats_dataframe(force_recompute:bool=False) -> pd.DataFrame|None:
    """Tries getting the stats dataframe. If it does not exist or force_recompute is set to true starts a thread that computes it and returns None. 
    \nCalling this with force_recompute will return None if the dataframe is still being computed!"""
    global stats_dataframe
    global status_stats_dataframe

    #The function called by the Thread object.
    #Computation of different measures of uncertainty is implemented here
    #TODO Add additional uncertainty measures
    def compute_stats_dataframe():
        global stats_dataframe
        global status_stats_dataframe

        model = load_model()
        data_loader = get_data_loader()
        model = model.cuda() #move model to gpu
        model.eval() #set to eval mode. disables dropout layers too.

        softmax = torch.nn.Softmax(dim=1)

        statistics = []
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
        stats_dataframe = df
        status_stats_dataframe = "Generated"

    if force_recompute: 
        if status_stats_dataframe != "Currently Generating": status_stats_dataframe = "Not Generated"
    match status_stats_dataframe:
        case "Not Generated":
            status_stats_dataframe = "Currently Generating" #Safety net for preventing restarting the Thread if it's still running
            Thread(target=compute_stats_dataframe).start()
            return None
        case "Currently Generating": return None #Safety net for preventing restarting the Thread if it's still running
        case "Generated": return stats_dataframe.to_json()

def get_images(image_indices: list[tuple[int,int]]) -> pd.DataFrame|None:
    """
    Parameters:
        image_indices: List of (batch_index, image_index) tuples
    Returns:
        DataFrame: JSON of pandas DataFrame with columns [batch_index, image_index, input_image, target_image, prediction_image, entropy_image] 
        or None if the dataloader is not yet initialized.

    Notes:
        Showcase for this function in lab_draft notebook. Strg+F for GETIMAGES
    """

    if not dataloader: return None

    #create a dict from the list for efficient enumeration (can't address directly with our dataloader)
    indices = defaultdict(list)
    for k,v in image_indices: indices[k].append(v)

    #setup model and utilities
    model = load_model()
    dataloader = get_data_loader()
    model = model.cuda() #move model to gpu
    model.eval() #set to eval mode. disables dropout layers too.
    softmax = torch.nn.Softmax(dim=1)

    position, batch_index, image_index, input_image, target_image, prediction_image, entropy_image = [],[],[],[],[],[],[]
    for i,batch in enumerate(dataloader):
        if i in indices.keys():
            images = batch["input"].cuda()
            for im_idx in indices[i]:
                position.append(image_indices.index((i,im_idx))) #Helper column for custom ordering
                batch_index.append(i)
                image_index.append(im_idx)
                input_image.append(batch["input"][im_idx].squeeze().numpy())
                target_image.append(batch["target"][im_idx].squeeze().numpy())
                
                #Get predictions and uncertainty images
                with torch.no_grad():
                    pred = model(images[im_idx:im_idx+1])
                    prob = softmax(pred)
                    prediction_image.append(prob.argmax(dim=1).squeeze().cpu().numpy())
                #TODO: Other uncertainties
                entropy = Categorical(prob.moveaxis(1,3)).entropy()
                entropy_image.append(entropy.squeeze().cpu().numpy())
        if i == max(indices): break #No need to keep enumerating if we have no remaining key denoting a later batch left in the dict
    
    #Prepare the results and return them
    data = zip(position, batch_index, image_index, input_image, target_image, prediction_image, entropy_image)
    images_df = pd.DataFrame(data, index=position, columns=["position", "batch_index", "image_index", "input_image", "target_image", "prediction_image", "entropy_image"]).sort_index()
    return images_df.drop(column="position").to_json() #don't need the helper column anymore

def get_image(batch_number, img_number):
    # img = test_loader.dataset[batch_number]['input'][img_number]
    dataloader = get_data_loader()
    for i, batch in enumerate(dataloader):
        if i == batch_number:
            break

    img = batch['input'][img_number]

    return img.tolist()

#Deprecated method: functionality is now handled by get_stats_dataframe. Should not be used by the api.
def compute_entropy_dataframe(model, data_loader=get_data_loader()):

    model = load_model()
    model = model.cuda() #move model to gpu
    model.eval() #set to eval mode. disables dropout layers too.

    softmax = torch.nn.Softmax(dim=1)

    statistics = []
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