from typing import Dict, List, Tuple
import os
import pickle 
import glob

import torch

import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm

from regym.thirdparty.ReferentialGym.ReferentialGym.utils import StreamHandler


VERBOSE = False 


class PubSubManager(object):
    def __init__(self, 
                 config={}, 
                 modules={}, 
                 pipelines={}, 
                 load_path=None, 
                 save_path=None,
                 verbose=False,
                 logger=None,
                 save_epoch_interval=None):
        self.verbose = verbose
        self.save_epoch_interval = save_epoch_interval

        self.load_path= load_path
        self.save_path = save_path

        self.config = config
        if load_path is not None:
            self.load_config(load_path)
        
        self.stream_handler = StreamHandler()
        self.stream_handler.register("losses_dict")
        self.stream_handler.register("logs_dict")
        self.stream_handler.register("signals")
        if load_path is not None:
            self.load_signals(load_path)
        
        # Register hyperparameters:
        for k,v in self.config.items():
            self.stream_handler.update(f"config:{k}", v)
        # Register modules:
        self.modules = modules
        if load_path is not None:
            self.load_modules(load_path)
        for k,m in self.modules.items():
            self.stream_handler.update(f"modules:{m.get_id()}:ref", m)

        if logger is not None:
            self.stream_handler.update("modules:logger:ref", logger)
        
        # Register pipelines:
        self.pipelines = pipelines
        if load_path is not None:
            self.load_pipelines(load_path)

    def save(self, path=None):
        if path is None:
            print("WARNING: no path provided for save. Saving in './temp_save/'.")
            path = './temp_save/'

        os.makedirs(path, exist_ok=True)

        self.save_config(path)
        self.save_modules(path)
        self.save_pipelines(path)
        self.save_signals(path)

        if self.verbose:
            print(f"Saving at {path}: OK.")

    def save_config(self, path):
        try:
            with open(os.path.join(path, "config.conf"), 'wb') as f:
                pickle.dump(self.config, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Exception caught while trying to save config: {e}")

    def save_modules(self, path):
        for module_id, module in self.modules.items():
            #try:
            if hasattr(module, "save"):
                module.save(path=path)
            else:
                torch.save(module, os.path.join(path,module_id+".pth"))
            #except Exception as e:
            #    print(f"Exception caught will trying to save module {module_id}: {e}")
                 

    def save_pipelines(self, path):
        try:
            with open(os.path.join(path, "pipelines.pipe"), 'wb') as f:
                pickle.dump(self.pipelines, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Exception caught while trying to save pipelines: {e}")

    def save_signals(self, path):
        try:
            with open(os.path.join(path, "signals.conf"), 'wb') as f:
                pickle.dump(self.stream_handler["signals"], f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Exception caught while trying to save signals: {e}")

    def load(self, path):
        self.load_config(path)
        self.load_modules(path)
        self.load_pipelines(path)
        self.load_signals(path)

        if self.verbose:
            print(f"Loading from {path}: OK.")


    def load_config(self, path):
        try:
            with open(os.path.join(path, "config.conf"), 'rb') as f:
                self.config = pickle.load(f)
        except Exception as e:
            print(f"Exception caught while trying to load config: {e}")

        if self.verbose:
            print(f"Loading config: OK.")

    def load_modules(self, path):
        modules_paths = glob.glob(os.path.join(path, "*.pth"))
        
        for module_path in modules_paths:
            module_id = module_path.split("/")[-1].split(".")[0]
            try:
                    self.modules[module_id] = torch.load(module_path)
            except Exception as e:
                print(f"Exception caught will trying to load module {module_path}: {e}")
        
        if self.verbose:
            print(f"Loading modules: OK.")
    
    def load_pipelines(self, path):
        try:
            with open(os.path.join(path, "pipelines.pipe"), 'rb') as f:
                self.pipelines.update(pickle.load(f))
        except Exception as e:
            print(f"Exception caught while trying to load pipelines: {e}")

        if self.verbose:
            print(f"Loading pipelines: OK.")

    def load_signals(self, path):
        try:
            with open(os.path.join(path, "signals.conf"), 'rb') as f:
                self.stream_handler.update("signals", pickle.load(f))
        except Exception as e:
            print(f"Exception caught while trying to load signals: {e}")

        if self.verbose:
            print(f"Loading signals: OK.")

    def train(self):
        iteration = -1
        
        while True:
            iteration += 1
            #pbar.update(1)

            self.stream_handler.update("signals:iteration", iteration)
            self.stream_handler.update("signals:global_it_step", iteration)
            
            for pipe_id, pipeline in self.pipelines.items():
                self.stream_handler.serve(pipeline)
            
            # Logging synchronously:
            ld = {}
            for k,v in self.stream_handler["logs_dict"].items():
                ld[k] = v 
                #print(k, type(v))
            for k,v in self.stream_handler["losses_dict"].items():
                ld[k] = v 
                #print(f"loss : {k} :", type(v))
            for k,v in self.stream_handler["signals"].items():
                ld[k] = v
                #print(f"signal : {k} :", type(v), v)
            wandb.log(ld, commit=True)

            self.stream_handler.reset("losses_dict")
            self.stream_handler.reset("logs_dict")    
            
            # TODO: define how to stop the loop?
            if self.stream_handler["signals:done_training"]:
                break
            # TODO: define how to make checkpoints:
            """
            if self.save_epoch_interval is not None\
             and epoch % self.save_epoch_interval == 0:
                self.save(path=self.save_path)
            """

        return
