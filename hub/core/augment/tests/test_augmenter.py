import hub
from hub.tests.common import get_dummy_data_path
from hub.api.dataset import Dataset
from hub.core.augment.transforms import *
from hub.core.augment.augment import Augmenter
from torch.utils.data import DataLoader
import albumentations as A
import pytest
import torch
import multiprocessing as mp


def test_augmenter_single():
    path = get_dummy_data_path("tests_augment/hub_classification")
    
    ds = hub.load(path)
    
    aug = Augmenter()
    aug.add_step(["images"], [A.Resize(25,25)])  
    
    loader = aug.augment(ds, ["images", "labels"]) 
    
    counter = 0   
    for _, sample in enumerate(loader):
        counter += 1
    
    dataset_length = len(ds)
    
    assert sample["images"].shape == torch.Size(torch.Size([1, 3, 25, 25]))
    assert isinstance(loader, DataLoader)
    assert counter == dataset_length, f"{counter} and {dataset_length}"
    assert sample["labels"].shape == torch.Size([1,1])

def test_augmenter_multiple():
    path = get_dummy_data_path("tests_augment/hub_classification")
    
    ds = hub.load(path)
    
    aug = Augmenter()
    aug.add_step(["images"], [A.Resize(25,25)])  
    aug.add_step(["images"], [A.Resize(25,25), A.VerticalFlip(p=1.0)])

    
    loader = aug.augment(ds, ["images", "labels"]) 
    
    counter = 0   
    for _, sample in enumerate(loader):
        counter += 1
    
    dataset_length = len(ds)
    
    assert sample["images"].shape == torch.Size(torch.Size([1, 3, 25, 25]))
    assert isinstance(loader, DataLoader)
    assert counter == 2*dataset_length, f"{counter} and {dataset_length}"
    assert sample["labels"].shape == torch.Size([1,1])


@pytest.mark.skipif(mp.cpu_count() == 1, reason="device lacks workers")
def test_augmenter_workers():
    path = get_dummy_data_path("tests_augment/hub_classification")
    
    ds = hub.load(path)
    
    aug = Augmenter()
    aug.add_step(["images"], [A.Resize(25,25)])  
    aug.add_step(["images"], [A.Resize(25,25), A.VerticalFlip(p=1.0)])

    
    loader = aug.augment(ds, ["images", "labels"], num_workers = 2) 
    
    counter = 0   
    for _, sample in enumerate(loader):
        counter += 1
    
    dataset_length = len(ds)
    
    assert counter == 2*dataset_length, f"{counter} and {dataset_length}"
    assert sample["images"].shape == torch.Size(torch.Size([1, 3, 25, 25]))
    assert sample["labels"].shape == torch.Size([1,1])
    
# @pytest.mark.skip(reason = "Not ready yet")
def test_augmenter_conditional():
    import dill
    path = get_dummy_data_path("tests_augment/hub_classification")
    
    ds = hub.load(path)
    
    aug = Augmenter()
    aug.add_step(["images"], [A.Resize(25,25)])  

    sample_condition = lambda sample: sample["labels"].item() == 0
    aug.add_step(["images"], [A.Resize(25,25), A.VerticalFlip(p=1.0)], sample_condition)

    
    loader = aug.augment(ds, ["images", "labels"])
    
    counter = 0   
    for _, sample in enumerate(loader):
        counter += 1
    
    dataset_length = len(ds)
    
    assert counter == 2*dataset_length-1, f"{counter} and {dataset_length}"
    assert sample["images"].shape == torch.Size(torch.Size([1, 3, 25, 25]))
    assert sample["labels"].shape == torch.Size([1,1])


def test_augmenter_batches():
    path = get_dummy_data_path("tests_augment/hub_classification")
    
    ds = hub.load(path)
    
    aug = Augmenter()
    aug.add_step(["images"], [A.Resize(25,25)])  
    aug.add_step(["images"], [A.Resize(25,25), A.VerticalFlip(p=1.0)])

    
    loader = aug.augment(ds, ["images", "labels"], batch_size = 2) 
    
    counter = 0   
    for _, sample in enumerate(loader):
        counter += 1
    
    dataset_length = len(ds)
    
    assert counter == dataset_length, f"{counter} and {dataset_length}"
    assert sample["images"].shape == torch.Size(torch.Size([2, 3, 25, 25]))
    assert sample["labels"].shape == torch.Size([2,1])


def test_augmenter_trivial_augspace():
    path = get_dummy_data_path("tests_augment/hub_classification")
    
    ds = hub.load(path)
    
    aug = Augmenter()
    aug.add_step(["images"], [A.Resize(25,25), TrivialAugment(include_transforms=[])])  
    loader = aug.augment(ds, ["images", "labels"]) 
    
    try:
        for _, sample in enumerate(loader):
            pass
        assert 0
    except:
        assert 1
    

    