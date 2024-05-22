from typing import Dict, List 

import torch
import numpy as np
import copy 

import ReferentialGym
from ReferentialGym.modules import Module


class AITAOModule(Module):
    def __init__(
        self, 
        id:str,
        config:Dict[str,object]
        ):
        """
        :param id: str defining the ID of the module.
        """

        input_stream_ids = {
            "epoch":"signals:epoch",
            "mode":"signals:mode",
            
            "it_step":"signals:it_step",
            # step in the communication round.
         
            "sample":"current_dataloader:sample",

            "end_of_dataset":"signals:end_of_dataset",  
            # boolean: whether the current batch/datasample is the last of the current dataset/mode.
            "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
            # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
            # is the last of the current sequence of repetition.
            "end_of_communication":"signals:end_of_communication",
            # boolean: whether the current communication round is the last of 
            # the current dialog.
            "dataset":"current_dataset:ref",

            "speaker_sentences_widx":"modules:current_speaker:sentences_widx", 
            "speaker_exp_indices":"current_dataloader:sample:speaker_indices", 
        }

        super(AITAOModule, self).__init__(
            id=id, 
            type="AITAOModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        self.indices = []
        self.speaker_sentences = {} #from dataset's idx to sentence.
        self.sentence2class = {}
        self.class_counter = 0
    
    # Adapted from: 
    # https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L37
    def _hashable_tensor(self, t):
        if isinstance(t, tuple):
            return t
        if isinstance(t, int):
            return t

        try:
            t = t.item()
        except:
            t = tuple(t.reshape(-1).tolist())
        
        return t


    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        
        :param input_streams_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
            - `'mode'`: String that defines what mode we are in, e.g. 'train' or 'test'. Those keywords are expected.
        """

        outputs_dict = {}

        epoch = input_streams_dict["epoch"]
        mode = input_streams_dict["mode"]
        it_step = input_streams_dict["it_step"]
        
        speaker_sentences_widx = input_streams_dict["speaker_sentences_widx"]
        speaker_exp_indices = input_streams_dict["speaker_exp_indices"]

        if "train" in mode and it_step == 0:
            # Record speaker's sentences:
            if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
                speaker_widx = input_streams_dict["speaker_sentences_widx"].cpu().detach()
                batch_size = speaker_widx.shape[0]
                speaker_widx = speaker_widx.reshape(batch_size, -1).numpy()
                indices = input_streams_dict["speaker_exp_indices"]
                indices = indices.cpu().detach().reshape(-1).numpy().tolist()
                for bidx, didx in enumerate(indices):
                    sentence = self._hashable_tensor(speaker_widx[bidx])
                    if sentence not in self.sentence2class:
                        self.sentence2class[sentence] = self.class_counter
                        self.class_counter += 1
                    self.speaker_sentences[didx] = sentence
                    self.indices.append(int(didx))

            # Is it the end of the epoch?
            end_of_epoch = all([input_streams_dict[key] for key in self.end_of_])
            
            if end_of_epoch:
                # update dataset:
                dataset = input_streams_dict["dataset"]
                ## assumes DualLabeledDataset...
                current_target_indices = dataset.train_classes
                current_mode2idx2class = dataset.mode2idx2class

                new_train_classes = {}
                new_mode2idx2class = {'train':{}, 'test':current_mode2idx2class['test']}
                missing_indices = set(range(len(dataset.datasets['train'])))
                missing_indices = missing_indices.difference(set(self.indices))
                complete_list_indices = self.indices+list(missing_indices)
                for didx in complete_list_indices:
                    # Due to ObverterSamplingScheme,
                    # it is likely that not all indices will be seen through out an epoch:
                    if didx in self.indices:
                        cl = self.sentence2class[self.speaker_sentences[didx]]
                    else:
                        cl = current_mode2idx2class['train'][didx]
                    if cl not in new_train_classes: new_train_classes[cl] = []
                    new_train_classes[cl].append(didx)
                    new_mode2idx2class['train'][didx] = cl

                dataset.train_classes = new_train_classes 

                test_idx_offset = len(dataset.datasets['train'])
                new_test_classes = {}
                for idx in range(len(dataset.datasets['test'])):
                    if hasattr(dataset.datasets['test'], 'getclass'):
                        cl = dataset.datasets['test'].getclass(idx)
                    else :
                        _, cl = dataset.datasets['test'][idx]
                    if cl not in new_test_classes: new_test_classes[cl] = []
                    new_test_classes[cl].append(test_idx_offset+idx)
                    new_mode2idx2class['test'][test_idx_offset+idx] = cl
                
                # Adding the train classes to the test classes so that we can sample
                # distractors from the train set:
                for cl in new_train_classes:
                    if cl not in new_test_classes:
                        new_test_classes[cl] = []
                    for idx in new_train_classes[cl]:
                        new_test_classes[cl].append(idx)
                        new_mode2idx2class['test'][idx] = cl
                
                dataset.test_classes = new_test_classes
                dataset.mode2idx2class = new_mode2idx2class

                # Reset:
                self.speaker_sentences = {} #from dataset's idx to sentence.
                self.indices = []
                self.sentence2class = {}
                self.class_counter = 0

        return outputs_dict

