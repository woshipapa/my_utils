import random 
import os
import torch 


class ControlRandom(object):
    random_data_path = "./random_data"
    random_data_dict = {}
    
    @classmethod
    def read_database(cls, random_data_path=None):
        if random_data_path:
            path_to_read = random_data_path
        else:
            path_to_read = ControlRandom.random_data_path

        for filename in os.listdir(path_to_read):
            if not filename.endwith(".pt"):
                continue 
            key = '.'.join(filename.split('.')[:-1])
            ControlRandom.random_data_dict[key] = torch.load(os.path.join(path_to_read,filename))
    
    @classmethod
    def set_random_data_dict(path: str):
        ControlRandom.random_data_path = path

    @classmethod
    def save_random(cls, tag, random_data):
        if not os.path.exists(ControlRandom.random_data_path):
            os.makedirs(ControlRandom.random_data_path)
        torch.save(random_data, os.path.join(ControlRandom.random_data_path, f"{tag}.pt"))


    @classmethod
    def load_random(cls, tag):
        return ControlRandom.random_data_dict.get(f"{tag}.pt")

    @classmethod
    def deal_with_random(cls, save: torch.bool, tag, random_data=None):
        if save:
            ControlRandom.save_random(tag, random_data)
            return None
        else:
            if not ControlRandom.random_data_dict:
                ControlRandom.read_database(ControlRandom.random_data_path)
            return ControlRandom.load_random(tag)
            
