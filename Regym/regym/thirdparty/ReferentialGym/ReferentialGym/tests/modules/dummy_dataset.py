from typing import Dict, List

import sys
import random
import numpy as np 
import argparse 
import copy

import torch
from torch.utils.data import Dataset 
from PIL import Image 


class DummyDataset(Dataset) :
    def __init__(self, train=True, transform=None, split_strategy=None, nbr_latents=10, nbr_values_per_latent=10) :
        '''
        :param split_strategy: str 
            e.g.: 'divider-10-offset-0'
        '''
        self.nbr_latents = nbr_latents
        self.nbr_values_per_latent = nbr_values_per_latent
        self.dataset_size = np.power(nbr_values_per_latent, nbr_latents)
        self.train = train
        self.transform = transform
        self.split_strategy = split_strategy


        # Load dataset
        self.imgs = [np.zeros((24,24,3))]*100
        self.latents_values = np.random.randint(
          low=0,
          high=self.nbr_values_per_latent,
          size=(100, self.nbr_latents)
        )
        self.latents_classes = copy.deepcopy(self.latents_values)
        self.test_latents_mask = np.zeros_like(self.latents_classes)
        self.targets = np.zeros(len(self.latents_classes)) #[random.randint(0, 10) for _ in self.imgs]
        for idx, latent_cls in enumerate(self.latents_classes):
            posX = latent_cls[-2]
            posY = latent_cls[-1]
            target = posX*self.nbr_values_per_latent+posY
            self.targets[idx] = target
            """
            self.targets[idx] = idx
            """

        if self.split_strategy is not None:
            raise NotImplementedError 
            strategy = self.split_strategy.split('-')
            if 'divider' in self.split_strategy and 'offset' in self.split_strategy:
                self.divider = int(strategy[1])
                assert(self.divider>0)
                self.offset = int(strategy[-1])
                assert(self.offset>=0 and self.offset<self.divider)
            elif 'combinatorial' in self.split_strategy:
                self.counter_test_threshold = int(strategy[0][len('combinatorial'):])
                # (default: 2) Specifies the threshold on the number of latent dimensions
                # whose values match a test value. Below this threshold, samples are used in training.
                # A value of 1 implies a basic train/test split that tests generalization to out-of-distribution values.
                # A value of 2 implies a train/test split that tests generalization to out-of-distribution pairs of values...
                # It implies that test value are encountered but never when combined with another test value.
                # It is a way to test for binary compositional generalization from well known stand-alone test values.
                # A value of 3 tests for ternary compositional generalization from well-known:
                # - stand-alone test values, and
                # - binary compositions of test values.
                
                '''
                With regards to designing axises as primitives:
                
                It implies that all the values on this latent axis are treated as test values
                when combined with a test value on any other latent axis.
                
                N.B.: it is not possible to test for out-of-distribution values in that context...
                N.B.1: It is required that the number of primitive latent axis be one less than
                        the counter_test_thershold, at most.

                A number of fillers along this primitive latent axis can then be specified in front
                of the FP pattern...
                Among the effective indices, those with an ordinal lower or equal to the number of
                filler allowed will be part of the training set.
                '''
                self.latent_dims = {}
                # self.strategy[0] : 'combinatorial'
                # 1: Y
                self.latent_dims['Y'] = {'size': 32}
                
                self.latent_dims['Y']['nbr_fillers'] = 0
                self.latent_dims['Y']['primitive'] = ('FP' in strategy[1])
                if self.latent_dims['Y']['primitive']:
                    self.latent_dims['Y']['nbr_fillers'] = int(strategy[1].split('FP')[0])

                self.latent_dims['Y']['position'] = 5
                # 2: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[2]:
                    strategy[2] = strategy[2].split('RemainderToUse')
                    self.latent_dims['Y']['remainder_use'] = int(strategy[2][1])
                    strategy[2] = strategy[2][0]
                else:
                    self.latent_dims['Y']['remainder_use'] = 0
                self.latent_dims['Y']['divider'] = int(strategy[2])
                # 3: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.
                if 'N' in strategy[3]:
                    self.latent_dims['Y']['untested'] = True
                    self.latent_dims['Y']['test_set_divider'] = (self.latent_dims['Y']['size']//self.latent_dims['Y']['divider'])+10
                elif 'E' in strategy[3]:  
                    self.latent_dims['Y']['test_set_size_sample_from_end'] = int(strategy[3][1:])
                elif 'S' in strategy[3]:  
                    self.latent_dims['Y']['test_set_size_sample_from_start'] = int(strategy[3][1:])
                else:
                    self.latent_dims['Y']['test_set_divider'] = int(strategy[3])

                # 4: X
                self.latent_dims['X'] = {'size': 32}
                
                self.latent_dims['X']['nbr_fillers'] = 0
                self.latent_dims['X']['primitive'] = ('FP' in strategy[4])
                if self.latent_dims['X']['primitive']:
                    self.latent_dims['X']['nbr_fillers'] = int(strategy[4].split('FP')[0])

                self.latent_dims['X']['position'] = 4
                # 5: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[5]:
                    strategy[5] = strategy[5].split('RemainderToUse')
                    self.latent_dims['X']['remainder_use'] = int(strategy[5][1])
                    strategy[5] = strategy[5][0]
                else:
                    self.latent_dims['X']['remainder_use'] = 0
                self.latent_dims['X']['divider'] = int(strategy[5])
                # 6: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[6]:
                    self.latent_dims['X']['untested'] = True
                    self.latent_dims['X']['test_set_divider'] = (self.latent_dims['X']['size']//self.latent_dims['X']['divider'])+10
                elif 'E' in strategy[6]:  
                    self.latent_dims['X']['test_set_size_sample_from_end'] = int(strategy[6][1:])
                elif 'S' in strategy[6]:  
                    self.latent_dims['X']['test_set_size_sample_from_start'] = int(strategy[6][1:])
                else:  
                    self.latent_dims['X']['test_set_divider'] = int(strategy[6])
                # 7: Orientation
                self.latent_dims['Orientation'] = {'size': 40}
                
                self.latent_dims['Orientation']['nbr_fillers'] = 0
                self.latent_dims['Orientation']['primitive'] = ('FP' in strategy[7])
                if self.latent_dims['Orientation']['primitive']:
                    self.latent_dims['Orientation']['nbr_fillers'] = int(strategy[7].split('FP')[0])

                self.latent_dims['Orientation']['position'] = 3
                # 8: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 10  
                if 'RemainderToUse' in strategy[8]:
                    strategy[8] = strategy[8].split('RemainderToUse')
                    self.latent_dims['Orientation']['remainder_use'] = int(strategy[8][1])
                    strategy[8] = strategy[8][0]
                else:
                    self.latent_dims['Orientation']['remainder_use'] = 0
                self.latent_dims['Orientation']['divider'] = int(strategy[8])
                # 9: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 and 10 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[9]:
                    self.latent_dims['Orientation']['untested'] = True
                    self.latent_dims['Orientation']['test_set_divider'] = (self.latent_dims['Orientation']['size']//self.latent_dims['Orientation']['divider'])+10
                elif 'E' in strategy[9]:  
                    self.latent_dims['Orientation']['test_set_size_sample_from_end'] = int(strategy[9][1:])
                elif 'S' in strategy[9]:  
                    self.latent_dims['Orientation']['test_set_size_sample_from_start'] = int(strategy[9][1:])
                else:
                    self.latent_dims['Orientation']['test_set_divider'] = int(strategy[9])
                
                # 10: Scale
                self.latent_dims['Scale'] = {'size': 6}
                
                self.latent_dims['Scale']['nbr_fillers'] = 0
                self.latent_dims['Scale']['primitive'] = ('FP' in strategy[10])
                if self.latent_dims['Scale']['primitive']:
                    self.latent_dims['Scale']['nbr_fillers'] = int(strategy[10].split('FP')[0])

                self.latent_dims['Scale']['position'] = 2
                # 11: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 6  
                if 'RemainderToUse' in strategy[11]:
                    strategy[11] = strategy[11].split('RemainderToUse')
                    self.latent_dims['Scale']['remainder_use'] = int(strategy[11][1])
                    strategy[11] = strategy[11][0]    
                else:
                    self.latent_dims['Scale']['remainder_use'] = 0
                self.latent_dims['Scale']['divider'] = int(strategy[11])
                # 12: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[12]:
                    self.latent_dims['Scale']['untested'] = True
                    self.latent_dims['Scale']['test_set_divider'] = (self.latent_dims['Scale']['size']//self.latent_dims['Scale']['divider'])+10
                elif 'E' in strategy[12]:  
                    self.latent_dims['Scale']['test_set_size_sample_from_end'] = int(strategy[12][1:])
                elif 'S' in strategy[12]:  
                    self.latent_dims['Scale']['test_set_size_sample_from_start'] = int(strategy[12][1:])
                else:
                    self.latent_dims['Scale']['test_set_divider'] = int(strategy[12])

                # 13: Shape
                self.latent_dims['Shape'] = {'size': 3}
                
                self.latent_dims['Shape']['nbr_fillers'] = 0
                self.latent_dims['Shape']['primitive'] = ('FP' in strategy[13])
                if self.latent_dims['Shape']['primitive']:
                    self.latent_dims['Shape']['nbr_fillers'] = int(strategy[13].split('FP')[0])

                self.latent_dims['Shape']['position'] = 1
                # 14: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 3  
                if 'RemainderToUse' in strategy[14]:
                    strategy[14] = strategy[14].split('RemainderToUse')
                    self.latent_dims['Shape']['remainder_use'] = int(strategy[14][1])
                    strategy[14] = strategy[14][0]    
                else:
                    self.latent_dims['Shape']['remainder_use'] = 0
                self.latent_dims['Shape']['divider'] = int(strategy[14])
                # 15: test_set_divider (default:3) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=3 => effective indices 3 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[15]:
                    self.latent_dims['Shape']['untested'] = True
                    self.latent_dims['Shape']['test_set_divider'] = (self.latent_dims['Shape']['size']//self.latent_dims['Shape']['divider'])+10
                else:  
                    self.latent_dims['Shape']['test_set_divider'] = int(strategy[15])
                
                # COLOR: TODO...

                nbr_primitives_and_tested = len([k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] or 'untested' not in self.latent_dims[k]])
                #assert(nbr_primitives_and_tested==self.counter_test_threshold)

        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        if self.split_strategy is None or 'divider' in self.split_strategy:
            self.indices = np.arange(100)
            """
            for idx in range(len(self.imgs)):
                if idx % self.divider == self.offset:
                    self.indices.append(idx)

            self.train_ratio = 0.8
            end = int(len(self.indices)*self.train_ratio)
            if self.train:
                self.indices = self.indices[:end]
            else:
                self.indices = self.indices[end:]
            """
            print(f"Split Strategy: {self.split_strategy} --> d {self.divider} / o {self.offset}")
            print(f"Dataset Size: {len(self.indices)} out of {self.dataset_size}: {100*len(self.indices)/self.dataset_size}%.")
        elif 'combinatorial' in self.split_strategy:
            raise NotImplementedError
            indices_latents = list(zip(range(self.latents_classes.shape[0]), self.latents_classes))
            for idx, latent_class in indices_latents:
                effective_test_threshold = self.counter_test_threshold
                counter_test = {}
                skip_it = False
                filler_forced_training = False
                for dim_name, dim_dict in self.latent_dims.items():
                    dim_class = latent_class[dim_dict['position']]
                    quotient = (dim_class+1)//dim_dict['divider']
                    remainder = (dim_class+1)%dim_dict['divider']
                    if remainder!=dim_dict['remainder_use']:
                        skip_it = True
                        break

                    if dim_dict['primitive']:
                        ordinal = quotient
                        if ordinal > dim_dict['nbr_fillers']:
                            effective_test_threshold -= 1

                    if 'test_set_divider' in dim_dict and quotient%dim_dict['test_set_divider']==0:
                        counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_end' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient > max_quotient-dim_dict['test_set_size_sample_from_end']:
                            counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_start' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient <= dim_dict['test_set_size_sample_from_start']:
                            counter_test[dim_name] = 1

                    if dim_name in counter_test:
                        self.test_latents_mask[idx, dim_dict['position']] = 1
                        
                if skip_it: continue


                if self.train:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        continue
                    else:
                        self.indices.append(idx)
                else:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        self.indices.append(idx)
                    else:
                        continue

            print(f"Split Strategy: {self.split_strategy}")
            print(self.latent_dims)
            print(f"Dataset Size: {len(self.indices)} out of 737280 : {100*len(self.indices)/737280}%.")
            

        #self.imgs = self.imgs[self.indices]
        self.latents_values = self.latents_values[self.indices]
        self.latents_classes = self.latents_classes[self.indices]
        
        self.latents_one_hot_encodings = np.eye(self.nbr_values_per_latent)[self.latents_classes.reshape(-1)]
        #self.latents_one_hot_encodings = self.latents_one_hot_encodings.reshape((-1, self.nbr_latents, self.nbr_values_per_latent))
        self.latents_one_hot_encodings = self.latents_one_hot_encodings.reshape((-1, self.nbr_latents*self.nbr_values_per_latent))

        self.test_latents_mask = self.test_latents_mask[self.indices]
        self.targets = self.targets[self.indices]
        
        print('Dataset loaded : OK.')
        
    def __len__(self) -> int:
        return 100 #len(self.indices)

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        target = self.targets[idx]
        return target

    def getlatentvalue(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        latent_value = self.latents_values[idx]
        return latent_value

    def getlatentclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        latent_class = self.latents_classes[idx]
        return latent_class

    def getlatentonehot(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        latent_one_hot_encoded = self.latents_one_hot_encodings[idx]
        return latent_one_hot_encoded

    def gettestlatentmask(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        test_latents_mask = self.test_latents_mask[idx]
        return test_latents_mask

    def sample_factors(self, num, random_state):
        """
        Sample a batch of factors Y.
        """
        #return random_state.randint(low=0, high=self.nbr_values_per_latent, size=(num, self.nbr_latents))
        # It turns out the random state is not really being updated apparently.

        # Therefore it was always sampling the same values...
        return np.random.randint(low=0, high=self.nbr_values_per_latent, size=(num, self.nbr_latents))
        
    def sample_observations_from_factors(self, factors, random_state):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        return factors

    def sample_latents_values_from_factors(self, factors, random_state):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        return factors

    def sample_latents_ohe_from_factors(self, factors, random_state):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        return factors

    def sample(self, num, random_state):
        """
        Sample a batch of factors Y and observations X.
        """
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def __getitem__(self, idx:int) -> Dict[str,torch.Tensor]:
        """
        :param idx: Integer index.

        :returns:
            sampled_d: Dict of:
                - `"experiences"`: Tensor of the sampled experiences.
                - `"exp_labels"`: List[int] consisting of the indices of the label to which the experiences belong.
                - `"exp_latents"`: Tensor representation of the latent of the experience in one-hot-encoded vector form.
                - `"exp_latents_values"`: Tensor representation of the latent of the experience in value form.
                - `"exp_latents_one_hot_encoded"`: Tensor representation of the latent of the experience in one-hot-encoded class form.
                - `"exp_test_latent_mask"`: Tensor that highlights the presence of test values, if any on each latent axis.
        """
        if idx >= len(self):
            idx = idx%len(self)
        #orig_idx = idx
        #idx = self.indices[idx]

        #img, target = self.dataset[idx]
        image = Image.fromarray((self.imgs[idx]*255).astype('uint8'))
        
        target = self.getclass(idx)
        latent_value = torch.from_numpy(self.getlatentvalue(idx))
        latent_class = torch.from_numpy(self.getlatentclass(idx))
        latent_one_hot_encoded = torch.from_numpy(self.getlatentonehot(idx))
        test_latents_mask = torch.from_numpy(self.gettestlatentmask(idx))

        if self.transform is not None:
            image = self.transform(image)
        
        sampled_d = {
            "experiences":image, 
            "exp_labels":target, 
            "exp_latents":latent_class, 
            "exp_latents_values":latent_value,
            "exp_latents_one_hot_encoded":latent_one_hot_encoded,
            "exp_test_latents_masks":test_latents_mask,
        }

        return sampled_d
