from hub.core.dataset import Dataset
from typing import List
from hub.core.transform.transform import ComputeFunction 
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import time

class Augmenter():    
  def __init__(self, transformation_dict=None):
    """
    Creates an Augmenter object. 
    
    Args:
      transformation_dict: (Optional) Give the transformation structure as input. It is a dictionary 
                            with tensors as keys and contains lists of augmentation strategies as values. 
                            Each augmentation strategy is a tupple of a list of augmentation_sequence and a condition variable
                            Structure: {"images": [augmentation_strategy_1, augmentation_strategy_2 ...]}
                                         where augmentation_strategy = (augmentation_sequence, sample_condition)
    """
    if transformation_dict!=None:
      self.transformation_dict = transformation_dict
    else:
      self.transformation_dict = {}

  def add_step(self, input_tensors , step_transform, sample_condition=None):
    """
    Adds a transformation_pipeline to each of the tensors in input_tensors.

    Args:
      input_tensors: List of tensors
      step_transform: The transformation sequence to be used to transform tensors.
      smaple_condition: A condition due to which decides whether the augmentation sequence will be used for the sample.
    """
    for tensor in input_tensors:
      if tensor not in self.transformation_dict.keys():
        self.transformation_dict[tensor] = [(step_transform, sample_condition)]
      else:
        self.transformation_dict[tensor].append((step_transform, sample_condition))

  def augment(self, ds, return_tensors, num_workers=1, batch_size=1):
    """
    Returns a Dataloader. Each sample in the dataloader contains a transformed tensor according to the defined steps.

    Args: 
      ds: Takes in a Hub dataset. 
      num_workers: The number of workers to use. 
      batch_size: Batch size to use.
    """
    
    transformation_info = [return_tensors, self.transformation_dict.copy()]
    return ds.pytorch(transform=transformation_info, multiple_transforms=True, num_workers=num_workers, batch_size=batch_size)