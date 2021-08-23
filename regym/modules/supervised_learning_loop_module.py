from typing import Dict, List 

import os
import math
import copy
import time
from tqdm import tqdm
import numpy as np

import regym
from tensorboardX import SummaryWriter

from regym.thirdparty.ReferentialGym.datasets import collate_dict_wrapper, PrioritizedSampler
from regym.modules.module import Module


def build_SupervisedLearningLoopModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None) -> Module:
    return SupervisedLearningLoopModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class SupervisedLearningLoopModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):
        
        default_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "stream_handler":"stream_handler",

            "iteration":"signals:iteration",
        }

        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids.keys():
                    input_stream_ids[default_id] = default_stream

        super(SupervisedLearningLoopModule, self).__init__(
            id=id,
            type="SupervisedLearningLoopModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
        
        # Dataset:
        if 'batch_size' not in self.config:
            self.config['batch_size'] = 32
        if 'dataloader_num_worker' not in self.config:
            self.config['dataloader_num_worker'] = 1

        print("Create dataloader: ...")
        
        self.datasets = self.config["datasets"]
        self.use_priority = self.config["use_priority"]
        self.logger = self.config["logger"]
        
        self.data_loaders = {}
        self.pbss = {}
        print("WARNING: 'dataloader_num_worker' hyperparameter has been de-activated.")
        for mode, dataset in self.datasets.items():
            if 'train' in mode and self.use_priority:
                capacity = len(dataset)
                
                pbs = PrioritizedSampler(
                    capacity=capacity,
                    batch_size=self.config['batch_size'],
                    logger=self.logger,
                )
                self.pbss[mode] = pbs
                self.data_loaders[mode] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config['batch_size'],
                    collate_fn=collate_dict_wrapper,
                    pin_memory=True,
                    #num_workers=self.config['dataloader_num_worker'],
                    sampler=pbs,
                )
            else:
                self.data_loaders[mode] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    collate_fn=collate_dict_wrapper,
                    pin_memory=True,
                    #num_workers=self.config['dataloader_num_worker']
                )
            
        print("Create dataloader: OK.")
        

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """

        self.stream_handler = input_streams_dict["stream_handler"]
        logger = input_streams_dict["logger"]
        nbr_epoch = self.config["nbr_epoch"]
        verbose_period = 1 

        print("Launching training: ...")

        it_datasamples = self.stream_handler['signals:it_datasamples']
        if it_datasamples is None:  it_datasamples = {mode:0 for mode in self.datasets} # counting the number of data sampled from dataloaders
        it_samples = self.stream_handler['signals:it_samples']
        if it_samples is None:  it_samples = {mode:0 for mode in self.datasets} # counting the number of multi-round
        it_steps = self.stream_handler['signals:it_steps']
        if it_steps is None:    it_steps = {mode:0 for mode in self.datasets} # taking into account multi round... counting the number of sample shown to the agents.
        
        init_curriculum_nbr_distractors = self.stream_handler["signals:curriculum_nbr_distractors"]
        if init_curriculum_nbr_distractors is None:
            init_curriculum_nbr_distractors = 1
        if self.config.get('use_curriculum_nbr_distractors', False):
            windowed_accuracy = 0.0
            window_count = 0
            for mode in self.datasets:
                self.datasets[mode].setNbrDistractors(init_curriculum_nbr_distractors,mode=mode)
            
        pbar = tqdm(total=nbr_epoch)
        if logger is not None:
            self.stream_handler.update("modules:logger:ref", logger)
        
        self.stream_handler.update("signals:use_cuda", self.config['use_cuda'])
        self.stream_handler.update("signals:update_count", 0)
                
        init_epoch = self.stream_handler["signals:epoch"]
        if init_epoch is None: 
            init_epoch = 0
        else:
            pbar.update(init_epoch)


        outputs_stream_dict = {}

        self.stream_handler.update("signals:done_supervised_learning_training", False) 
        epoch = init_epoch-1

        while self.stream_handler["signals:done_supervised_learning_training"]:
        #for epoch in range(init_epoch,nbr_epoch):
            epoch += 1
            if epoch > nbr_epoch:
                break

            self.stream_handler.update("signals:epoch", epoch)
            pbar.update(1)
            for it_dataset, (mode, data_loader) in enumerate(data_loaders.items()):
                self.stream_handler.update("current_dataset:ref", self.datasets[mode])
                self.stream_handler.update("signals:mode", mode)
                
                end_of_epoch_dataset = (it_dataset==len(data_loaders)-1)
                self.stream_handler.update("signals:end_of_epoch_dataset", end_of_epoch_dataset)
                
                nbr_experience_repetition = 1
                if 'nbr_experience_repetition' in self.config\
                    and 'train' in mode:
                    nbr_experience_repetition = self.config['nbr_experience_repetition']

                for idx_stimulus, sample in enumerate(data_loader):
                    end_of_dataset = (idx_stimulus==len(data_loader)-1)
                    self.stream_handler.update("signals:end_of_dataset", end_of_dataset)
                    it_datasamples[mode] += 1
                    it_datasample = it_datasamples[mode]
                    self.stream_handler.update("signals:it_datasamples", it_datasamples)
                    self.stream_handler.update("signals:global_it_datasample", it_datasample)
                    self.stream_handler.update("signals:it_datasample", idx_stimulus)
                    it = it_datasample


                    if self.config['use_cuda']:
                        sample = sample.cuda()

                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    
                    for it_rep in range(nbr_experience_repetition):
                        it_samples[mode] += 1
                        it_sample = it_samples[mode]
                        self.stream_handler.update("signals:it_samples", it_samples)
                        self.stream_handler.update("signals:global_it_sample", it_sample)
                        self.stream_handler.update("signals:it_sample", it_rep)
                        end_of_repetition_sequence = (it_rep==nbr_experience_repetition-1)
                        self.stream_handler.update("signals:end_of_repetition_sequence", end_of_repetition_sequence)
                        
                        # TODO: implement a multi_round_communicatioin module ?
                        for idx_round in range(self.config['nbr_communication_round']):
                            it_steps[mode] += 1
                            it_step = it_steps[mode]
                            
                            self.stream_handler.update("signals:it_steps", it_steps)
                            self.stream_handler.update("signals:global_it_step", it_step)
                            self.stream_handler.update("signals:it_step", idx_round)
                            
                            end_of_communication = (idx_round==self.config['nbr_communication_round']-1)
                            self.stream_handler.update("signals:end_of_communication", end_of_communication)
                            
                            multi_round = True
                            if end_of_communication:
                                multi_round = False
                            self.stream_handler.update("signals:multi_round", multi_round)
                            
                            #self.stream_handler.update('current_dataloader:sample', sample)
                            outputs_stream_dict["current_dataloader:sample"] = sample
                            
                            yield outputs_stream_dict
                            
                            """
                            for pipe_id, pipeline in self.pipelines.items():
                                if "referential_game" in pipe_id: 
                                    self.stream_handler.serve(pipeline)
                            """
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        
                        """
                        for pipe_id, pipeline in self.pipelines.items():
                            if "referential_game" not in pipe_id:
                                self.stream_handler.serve(pipeline)
                        """

                        losses = self.stream_handler["losses_dict"]

                        if self.use_priority and mode in self.pbss:
                            batched_loss = sum([l for l in losses.values()]).detach().cpu().numpy()
                            if len(batched_loss):
                                self.pbss[mode].update_batch(batched_loss)

                        loss = sum( [l.mean() for l in losses.values()])
                        logs_dict = self.stream_handler["logs_dict"]
                        acc_keys = [k for k in logs_dict.keys() if '/referential_game_accuracy' in k]
                        if len(acc_keys):
                            acc = logs_dict[acc_keys[-1]].mean()

                        if verbose_period is not None and idx_stimulus % verbose_period == 0:
                            descr = f"Epoch {epoch+1} :: {mode} Iteration {idx_stimulus+1}/{len(data_loader)}"
                            if isinstance(loss, torch.Tensor): loss = loss.item()
                            descr += f" (Rep:{it_rep+1}/{nbr_experience_repetition}):: Loss {it+1} = {loss}"
                            pbar.set_description_str(descr)
                        
                        self.stream_handler.reset("losses_dict")
                        self.stream_handler.reset("logs_dict")

                        '''
                        if logger is not None:
                            if self.config['with_utterance_penalization'] or self.config['with_utterance_promotion']:
                                import ipdb; ipdb.set_trace()
                                for widx in range(self.config['vocab_size']+1):
                                    logger.add_scalar("{}/Word{}Counts".format(mode,widx), speaker_outputs['speaker_utterances_count'][widx], it_step)
                                logger.add_scalar("{}/OOVLoss".format(mode), speaker_losses['oov_loss'][-1].mean().item(), it_step)
                            
                            if 'with_mdl_principle' in self.config and self.config['with_mdl_principle']:
                                logger.add_scalar("{}/MDLLoss".format(mode), speaker_losses['mdl_loss'][-1].mean().item(), it_step)
                        '''    
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        
                        # TODO: CURRICULUM ON DISTRATORS as a module that handles the current dataloader reference....!!
                        if 'use_curriculum_nbr_distractors' in self.config\
                            and self.config['use_curriculum_nbr_distractors']:
                            nbr_distractors = self.datasets[mode].getNbrDistractors(mode=mode)
                            self.stream_handler.update("signals:curriculum_nbr_distractors", nbr_distractors)
                            logger.add_scalar( "{}/CurriculumNbrDistractors".format(mode), nbr_distractors, it_step)
                            logger.add_scalar( "{}/CurriculumWindowedAcc".format(mode), windowed_accuracy, it_step)
                        
                        
                        # TODO: make this a logger module:
                        """
                        if 'current_speaker' in self.modules and 'current_listener' in self.modules:
                            prototype_speaker = self.stream_handler["modules:current_speaker:ref_agent"]
                            prototype_listener = self.stream_handler["modules:current_listener:ref_agent"]
                            image_save_path = logger.path 
                            if prototype_speaker is not None and hasattr(prototype_speaker,'VAE') and idx_stimulus % 4 == 0:
                                query_vae_latent_space(prototype_speaker.VAE, 
                                                       sample=sample['speaker_experiences'],
                                                       path=image_save_path,
                                                       test=('test' in mode),
                                                       full=('test' in mode),
                                                       idxoffset=it_rep+idx_stimulus*self.config['nbr_experience_repetition'],
                                                       suffix='speaker',
                                                       use_cuda=True)
                                
                            if prototype_listener is not None and hasattr(prototype_listener,'VAE') and idx_stimulus % 4 == 0:
                                query_vae_latent_space(prototype_listener.VAE, 
                                                       sample=sample['listener_experiences'],
                                                       path=image_save_path,
                                                       test=('test' in mode),
                                                       full=('test' in mode),
                                                       idxoffset=idx_stimulus,
                                                       suffix='listener')
                        """     
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//

                    # TODO: many parts everywhere, do not forget them all : CURRICULUM ON DISTRACTORS...!!!
                    if 'train' in mode\
                        and 'use_curriculum_nbr_distractors' in self.config\
                        and self.config['use_curriculum_nbr_distractors']:
                        nbr_distractors = self.datasets[mode].getNbrDistractors(mode=mode)
                        windowed_accuracy = (windowed_accuracy*window_count+acc.item())
                        window_count += 1
                        windowed_accuracy /= window_count
                        if windowed_accuracy > 75 and window_count > self.config['curriculum_distractors_window_size'] and nbr_distractors < self.config['nbr_distractors'][mode]:
                            windowed_accuracy = 0
                            window_count = 0
                            for mode in self.datasets:
                                self.datasets[mode].setNbrDistractors(self.datasets[mode].getNbrDistractors(mode=mode)+1, mode=mode)
                    # //------------------------------------------------------------//

                if logger is not None:
                    logger.switch_epoch()
                    
                # //------------------------------------------------------------//
                # //------------------------------------------------------------//
                # //------------------------------------------------------------//
            
            """
            if self.save_epoch_interval is not None\
             and epoch % self.save_epoch_interval == 0:
                self.save(path=self.save_path)
            """

            # //------------------------------------------------------------//
            # //------------------------------------------------------------//
            # //------------------------------------------------------------//

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------// 

        outputs_stream_dict["signals:done_supervised_learning_training"] = True 
        
        return outputs_stream_dict
            


    
