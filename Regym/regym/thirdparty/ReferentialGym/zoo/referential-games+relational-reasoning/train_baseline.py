import sys
import random
import numpy as np 
import argparse 
import copy

import ReferentialGym

import torch
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision
import torchvision.transforms as T 


def main():
  parser = argparse.ArgumentParser(description='LSTM CNN Agents: ST-GS Language Emergence.')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--use_cuda', action='store_true', default=False)
  parser.add_argument('--dataset', type=str, 
    choices=['Sort-of-CLEVR',
             'tiny-Sort-of-CLEVR',
             'XSort-of-CLEVR',
             'tiny-XSort-of-CLEVR',
             ], 
    help='dataset to train on.',
    default='XSort-of-CLEVR')
  parser.add_argument('--arch', type=str, 
    choices=['Santoro2017-SoC-CNN',
             'Santoro2017-CLEVR-CNN',
             ], 
    help='model architecture to train',
    default="Santoro2017-CLEVR-CNN")
  parser.add_argument('--graphtype', type=str,
    choices=['straight_through_gumbel_softmax',
             'reinforce',
             'baseline_reduced_reinforce',
             'normalized_reinforce',
             'baseline_reduced_normalized_reinforce',
             'max_entr_reinforce',
             'baseline_reduced_normalized_max_entr_reinforce',
             'argmax_reinforce',
             'obverter'],
    help='type of graph to use during training of the speaker and listener.',
    default='straight_through_gumbel_softmax')
  parser.add_argument('--max_sentence_length', type=int, default=15)
  parser.add_argument('--vocab_size', type=int, default=25)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--epoch', type=int, default=1600)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--mini_batch_size', type=int, default=64)
  parser.add_argument('--dropout_prob', type=float, default=0.0)
  parser.add_argument('--nbr_train_dataset_repetition', type=int, default=1)
  parser.add_argument('--nbr_test_dataset_repetition', type=int, default=1)
  parser.add_argument('--nbr_test_distractors', type=int, default=1)
  parser.add_argument('--nbr_train_distractors', type=int, default=1)
  parser.add_argument('--resizeDim', default=75, type=int,help='input image resize')
  parser.add_argument('--shared_architecture', action='store_true', default=False)
  parser.add_argument('--homoscedastic_multitasks_loss', action='store_true', default=False)
  parser.add_argument('--use_feat_converter', action='store_true', default=False)
  parser.add_argument('--detached_heads', action='store_true', default=False)
  parser.add_argument('--test_id_analogy', action='store_true', default=False)
  parser.add_argument('--train_test_split_strategy', type=str, 
    choices=['combinatorial2-Y-2-8-X-2-8-Orientation-40-N-Scale-6-N-Shape-3-N', # Exp : DoRGsFurtherDise interweaved split simple XY normal             
            ],
    help='train/test split strategy',
    default='combinatorial2-Y-2-8-X-2-8-Orientation-40-N-Scale-6-N-Shape-3-N')
  parser.add_argument('--fast', action='store_true', default=False, 
    help='Disable the deterministic CuDNN. It is likely to make the computation faster.')
  
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  # VAE Hyperparameters:
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  parser.add_argument('--vae_detached_featout', action='store_true', default=False)

  parser.add_argument('--vae_lambda', type=float, default=1.0)
  parser.add_argument('--vae_use_mu_value', action='store_true', default=False)
  
  parser.add_argument('--vae_nbr_latent_dim', type=int, default=128)
  parser.add_argument('--vae_decoder_nbr_layer', type=int, default=3)
  parser.add_argument('--vae_decoder_conv_dim', type=int, default=32)
  
  parser.add_argument('--vae_gaussian', action='store_true', default=False)
  parser.add_argument('--vae_gaussian_sigma', type=float, default=0.25)
  
  parser.add_argument('--vae_beta', type=float, default=1.0)
  parser.add_argument('--vae_factor_gamma', type=float, default=0.0)
  
  parser.add_argument('--vae_constrained_encoding', action='store_true', default=False)
  parser.add_argument('--vae_max_capacity', type=float, default=1e3)
  parser.add_argument('--vae_nbr_epoch_till_max_capacity', type=int, default=10)

  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  
  
  args = parser.parse_args()
  print(args)


  gaussian = args.vae_gaussian 
  vae_observation_sigma = args.vae_gaussian_sigma
  
  vae_beta = args.vae_beta 
  factor_vae_gamma = args.vae_factor_gamma
  
  vae_constrainedEncoding = args.vae_constrained_encoding
  maxCap = args.vae_max_capacity #1e2
  nbrepochtillmaxcap = args.vae_nbr_epoch_till_max_capacity

  monet_gamma = 5e-1
  
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  
  seed = args.seed 

  # Following: https://pytorch.org/docs/stable/notes/randomness.html
  torch.manual_seed(seed)
  if hasattr(torch.backends, 'cudnn') and not(args.fast):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  np.random.seed(seed)
  random.seed(seed)
  # # Hyperparameters:

  nbr_epoch = args.epoch
  
  cnn_feature_size = 512 # 128 512 #1024
  # Except for VAEs...!
  
  stimulus_resize_dim = args.resizeDim #64 #28
  
  normalize_rgb_values = False 
  
  rgb_scaler = 1.0 #255.0
  from ReferentialGym.datasets.utils import ResizeNormalize
  transform = ResizeNormalize(size=stimulus_resize_dim, 
                              normalize_rgb_values=normalize_rgb_values,
                              rgb_scaler=rgb_scaler)

  transform_degrees = 45
  transform_translate = (0.25, 0.25)

  multi_head_detached = args.detached_heads 

  rg_config = {
      "observability":            "partial",
      "max_sentence_length":      args.max_sentence_length, #5,
      "nbr_communication_round":  1,
      "nbr_distractors":          {'train':args.nbr_train_distractors, 'test':args.nbr_test_distractors},
      "distractor_sampling":      "uniform",#"similarity-0.98",#"similarity-0.75",
      # Default: use 'similarity-0.5'
      # otherwise the emerging language 
      # will have very high ambiguity...
      # Speakers find the strategy of uttering
      # a word that is relevant to the class/label
      # of the target, seemingly.  
      
      "descriptive":              False,
      "descriptive_target_ratio": 0.97, 
      # Default: 1-(1/(nbr_distractors+2)), 
      # otherwise the agent find the local minimum
      # where it only predicts 'no-target'...

      "object_centric":           False,
      "nbr_stimulus":             1,

      "graphtype":                args.graphtype,
      "tau0":                     0.2,
      "gumbel_softmax_eps":       1e-6,
      "vocab_size":               args.vocab_size,
      "symbol_embedding_size":    256, #64

      "agent_architecture":       args.arch, #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
      "agent_learning":           'learning',  #'transfer_learning' : CNN's outputs are detached from the graph...
      "agent_loss_type":          'Hinge', #'NLL'

      "cultural_pressure_it_period": None,
      "cultural_speaker_substrate_size":  1,
      "cultural_listener_substrate_size":  1,
      "cultural_reset_strategy":  "oldestL", # "uniformSL" #"meta-oldestL-SGD"
      "cultural_reset_meta_learning_rate":  1e-3,

      "iterated_learning_scheme": False,
      "iterated_learning_period": 200,

      "obverter_stop_threshold":  0.95,  #0.0 if not in use.
      "obverter_nbr_games_per_round": 2,

      "obverter_least_effort_loss": False,
      "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],

      "batch_size":               args.batch_size,
      "dataloader_num_worker":    4,
      "stimulus_depth_dim":       1 if 'dSprites' in args.dataset else 3,
      "stimulus_resize_dim":      stimulus_resize_dim, 
      
      "learning_rate":            args.lr, #1e-3,
      "adam_eps":                 1e-8,
      "dropout_prob":             args.dropout_prob,
      "embedding_dropout_prob":   0.8,
      
      "with_gradient_clip":       False,
      "gradient_clip":            1e0,
      
      "use_homoscedastic_multitasks_loss": args.homoscedastic_multitasks_loss,

      "use_feat_converter":       args.use_feat_converter,

      "use_curriculum_nbr_distractors": False,
      "curriculum_distractors_window_size": 25, #100,

      "unsupervised_segmentation_factor": None, #1e5
      "nbr_experience_repetition":  1,
      "nbr_dataset_repetition":  {'test':args.nbr_test_dataset_repetition, 'train':args.nbr_train_dataset_repetition},

      "with_utterance_penalization":  False,
      "with_utterance_promotion":     False,
      "utterance_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
                                                # The greater this value, the greater the loss/cost.
      "utterance_factor":    1e-2,

      "with_speaker_entropy_regularization":  False,
      "with_listener_entropy_regularization":  False,
      "entropy_regularization_factor":    -1e-2,

      "with_mdl_principle":       False,
      "mdl_principle_factor":     5e-2,

      "with_weight_maxl1_loss":   False,

      "with_grad_logging":        False,
      "use_cuda":                 args.use_cuda,
  
      # "train_transform":          T.Compose([T.RandomAffine(degrees=transform_degrees, 
      #                                                       translate=transform_translate, 
      #                                                       scale=None, 
      #                                                       shear=None, 
      #                                                       resample=False, 
      #                                                       fillcolor=0),
      #                                         transform]),

      # "test_transform":           T.Compose([T.RandomAffine(degrees=transform_degrees, 
      #                                                      translate=transform_translate, 
      #                                                      scale=None, 
      #                                                      shear=None, 
      #                                                      resample=False, 
      #                                                      fillcolor=0),
      #                                         transform]),
  
      "train_transform":            transform,
      "test_transform":             transform,
  }

  ## Train set:
  train_split_strategy = args.train_test_split_strategy
  test_split_strategy = train_split_strategy
  
  ## Agent Configuration:
  agent_config = dict()
  agent_config['use_cuda'] = rg_config['use_cuda']
  agent_config['homoscedastic_multitasks_loss'] = rg_config['use_homoscedastic_multitasks_loss']
  agent_config['use_feat_converter'] = rg_config['use_feat_converter']
  agent_config['max_sentence_length'] = rg_config['max_sentence_length']
  agent_config['nbr_distractors'] = rg_config['nbr_distractors']['train'] if rg_config['observability'] == 'full' else 0
  agent_config['nbr_stimulus'] = rg_config['nbr_stimulus']
  agent_config['nbr_communication_round'] = rg_config['nbr_communication_round']
  agent_config['descriptive'] = rg_config['descriptive']
  agent_config['gumbel_softmax_eps'] = rg_config['gumbel_softmax_eps']
  agent_config['agent_learning'] = rg_config['agent_learning']

  agent_config['symbol_embedding_size'] = rg_config['symbol_embedding_size']

  # Recurrent Convolutional Architecture:
  agent_config['architecture'] = rg_config['agent_architecture']
  agent_config['dropout_prob'] = rg_config['dropout_prob']
  agent_config['embedding_dropout_prob'] = rg_config['embedding_dropout_prob']
  
  if 'Santoro2017-SoC-CNN' in agent_config['architecture']:
    # For a fair comparison between CNN an VAEs:
    # the CNN is augmented with one final FC layer reducing to the latent space shape.
    # Need to use feat converter too:
    #rg_config['use_feat_converter'] = True 
    #agent_config['use_feat_converter'] = True 
    
    # Otherwise, the VAE alone may be augmented:
    # This approach assumes that the VAE latent dimension size
    # is acting as a prior which is part of the comparison...
    rg_config['use_feat_converter'] = False
    agent_config['use_feat_converter'] = False
    
    agent_config['cnn_encoder_channels'] = ['BN32','BN64','BN128','BN256']
    agent_config['cnn_encoder_kernels'] = [4,4,4,4]
    agent_config['cnn_encoder_strides'] = [2,2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1,1]
    agent_config['cnn_encoder_fc_hidden_units'] = [] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    # Otherwise:
    cnn_feature_size = 100
    agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config['cnn_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['feat_converter_output_size'] = cnn_feature_size

    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128

    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*cnn_feature_size
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1

  if 'Santoro2017-CLEVR-CNN' in agent_config['architecture']:
    # For a fair comparison between CNN an VAEs:
    # the CNN is augmented with one final FC layer reducing to the latent space shape.
    # Need to use feat converter too:
    #rg_config['use_feat_converter'] = True 
    #agent_config['use_feat_converter'] = True 
    
    # Otherwise, the VAE alone may be augmented:
    # This approach assumes that the VAE latent dimension size
    # is acting as a prior which is part of the comparison...
    rg_config['use_feat_converter'] = False
    agent_config['use_feat_converter'] = False
    
    agent_config['cnn_encoder_channels'] = ['BN24','BN24','BN24','BN24']
    agent_config['cnn_encoder_kernels'] = [4,4,4,4]
    agent_config['cnn_encoder_strides'] = [2,2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1,1]
    agent_config['cnn_encoder_fc_hidden_units'] = [] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    # Otherwise:
    cnn_feature_size = 100
    agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config['cnn_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['feat_converter_output_size'] = cnn_feature_size

    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128

    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*cnn_feature_size
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1




  save_path = f"./{args.dataset}+DualLabeled/{'Attached' if not(multi_head_detached) else 'Detached'}Heads"
  save_path += f"/{nbr_epoch}Ep_Emb{rg_config['symbol_embedding_size']}_CNN{cnn_feature_size}to{args.vae_nbr_latent_dim}"
  if args.shared_architecture:
    save_path += "/shared_architecture"
  save_path += f"/TrainNOTF_TestNOTF/"
  save_path += f"Dropout{rg_config['dropout_prob']}_DPEmb{rg_config['embedding_dropout_prob']}"
  save_path += f"_BN_{rg_config['agent_learning']}/"
  save_path += f"{rg_config['agent_loss_type']}"
  
  save_path += f"/OBS{rg_config['stimulus_resize_dim']}X{rg_config['stimulus_depth_dim']}C"
  
  if rg_config['use_curriculum_nbr_distractors']:
    save_path += f"+W{rg_config['curriculum_distractors_window_size']}Curr"
  if rg_config['with_utterance_penalization']:
    save_path += "+Tau-10-OOV{}PenProb{}".format(rg_config['utterance_factor'], rg_config['utterance_oov_prob'])  
  if rg_config['with_utterance_promotion']:
    save_path += "+Tau-10-OOV{}ProProb{}".format(rg_config['utterance_factor'], rg_config['utterance_oov_prob'])  
  
  if rg_config['with_gradient_clip']:
    save_path += '+ClipGrad{}'.format(rg_config['gradient_clip'])
  
  if rg_config['with_speaker_entropy_regularization']:
    save_path += 'SPEntrReg{}'.format(rg_config['entropy_regularization_factor'])
  if rg_config['with_listener_entropy_regularization']:
    save_path += 'LSEntrReg{}'.format(rg_config['entropy_regularization_factor'])
  
  if rg_config['iterated_learning_scheme']:
    save_path += '-ILM{}+ListEntrReg'.format(rg_config['iterated_learning_period'])
  
  if rg_config['with_mdl_principle']:
    save_path += '-MDL{}'.format(rg_config['mdl_principle_factor'])
  
  if rg_config['cultural_pressure_it_period'] != 'None':  
    save_path += '-S{}L{}-{}-Reset{}'.\
      format(rg_config['cultural_speaker_substrate_size'], 
      rg_config['cultural_listener_substrate_size'],
      rg_config['cultural_pressure_it_period'],
      rg_config['cultural_reset_strategy']+str(rg_config['cultural_reset_meta_learning_rate']) if 'meta' in rg_config['cultural_reset_strategy'] else rg_config['cultural_reset_strategy'])
  
  save_path += '-{}{}CulturalDiffObverter{}-{}GPR-SEED{}-{}-obs_b{}_lr{}-{}-tau0-{}-{}DistrTrain{}Test{}-stim{}-vocab{}over{}_{}{}'.\
    format(
    'ObjectCentric' if rg_config['object_centric'] else '',
    'Descriptive{}'.format(rg_config['descriptive_target_ratio']) if rg_config['descriptive'] else '',
    rg_config['obverter_stop_threshold'],
    rg_config['obverter_nbr_games_per_round'],
    seed,
    rg_config['observability'], 
    rg_config['batch_size'], 
    rg_config['learning_rate'],
    rg_config['graphtype'], 
    rg_config['tau0'], 
    rg_config['distractor_sampling'],
    *rg_config['nbr_distractors'].values(), 
    rg_config['nbr_stimulus'], 
    rg_config['vocab_size'], 
    rg_config['max_sentence_length'], 
    rg_config['agent_architecture'],
    f"/{'Detached' if args.vae_detached_featout else ''}beta{vae_beta}-factor{factor_vae_gamma}" if 'BetaVAE' in rg_config['agent_architecture'] else '')

  if 'MONet' in rg_config['agent_architecture'] or 'BetaVAE' in rg_config['agent_architecture']:
    save_path += f"beta{vae_beta}-factor{factor_vae_gamma}-gamma{monet_gamma}-sigma{vae_observation_sigma}" if 'MONet' in rg_config['agent_architecture'] else ''
    save_path += f"CEMC{maxCap}over{nbrepochtillmaxcap}" if vae_constrainedEncoding else ''
    save_path += f"UnsupSeg{rg_config['unsupervised_segmentation_factor']}Rep{rg_config['nbr_experience_repetition']}" if rg_config['unsupervised_segmentation_factor'] is not None else ''
    save_path += f"LossVAECoeff{args.vae_lambda}_{'UseMu' if args.vae_use_mu_value else ''}"

  if rg_config['use_feat_converter']:
    save_path += f"+FEATCONV"
  
  if rg_config['use_homoscedastic_multitasks_loss']:
    save_path += '+Homo'
  
  if 'reinforce' in args.graphtype:
    save_path += f'/REINFORCE_EntropyCoeffNeg1m3/UnnormalizedDetLearningSignalHavrylovLoss/NegPG/'

  save_path += f"/BASELINE_ALONE/"
  
  if args.test_id_analogy:
    save_path += 'withAnalogyTest/'
  else:
    save_path += 'NoAnalogyTest/'
  
  save_path += f'DatasetRepTrain{args.nbr_train_dataset_repetition}Test{args.nbr_test_dataset_repetition}'
  
  rg_config['save_path'] = save_path
  
  print(save_path)

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  
  # # Agents
  batch_size = 4
  nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']['train']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  # # Dataset:
  need_dict_wrapping = {}

  if 'XSort-of-CLEVR' in args.dataset:
    if 'tiny' in args.dataset:
      generate=True 
      dataset_size=1000
      test_size=200
      img_size=75
      object_size=5
      nb_objects=6
      test_id_analogy = args.test_id_analogy
      test_id_analogy_threshold = 3
    else:
      generate=True 
      dataset_size=10000
      test_size=2000
      img_size=75
      object_size=5
      nb_objects=6
      test_id_analogy = args.test_id_analogy
      test_id_analogy_threshold = 3
      
    root = './datasets/ext-sort-of-CLEVR-dataset'
    root += f'-{dataset_size}'
    root += f'-imgS{img_size}-objS{object_size}-obj{nb_objects}'
    
    train_dataset = ReferentialGym.datasets.XSortOfCLEVRDataset(root=root, 
      train=True, 
      transform=rg_config['train_transform'],
      generate=generate,
      dataset_size=dataset_size,
      test_size=test_size,
      img_size=img_size,
      object_size=object_size,
      nb_objects=nb_objects,
      test_id_analogy=test_id_analogy,
      test_id_analogy_threshold=test_id_analogy_threshold)
    
    test_dataset = ReferentialGym.datasets.XSortOfCLEVRDataset(root=root, 
      train=False, 
      transform=rg_config['test_transform'],
      generate=False,
      dataset_size=dataset_size,
      test_size=test_size,
      img_size=img_size,
      object_size=object_size,
      nb_objects=nb_objects,
      test_id_analogy=test_id_analogy,
      test_id_analogy_threshold=test_id_analogy_threshold)

    n_answers = train_dataset.answer_size
    if test_id_analogy:
      nb_questions = 3
    else:
      nb_questions = nb_objects

    nb_r_qs = 7
    nb_nr_qs = 5 
    

  elif 'Sort-of-CLEVR' in args.dataset:
    if 'tiny' in args.dataset:
      generate=True 
      dataset_size=1000
      test_size=200
      img_size=75
      object_size=5
      nb_objects=6
      test_id_analogy = args.test_id_analogy
      test_id_analogy_threshold = 3
    else:
      generate=True 
      dataset_size=10000
      test_size=2000
      img_size=75
      object_size=5
      nb_objects=6
      test_id_analogy = args.test_id_analogy
      test_id_analogy_threshold = 3

    nb_r_qs = 3
    nb_nr_qs = 3 

    n_answers = 4+nb_objects
    if test_id_analogy:
      nb_questions = 3
    else:
      nb_questions = nb_objects

    root = './datasets/sort-of-CLEVR-dataset'
    root += f'-{dataset_size}'
    root += f'-imgS{img_size}-objS{object_size}-obj{nb_objects}'
    
    train_dataset = ReferentialGym.datasets.SortOfCLEVRDataset(root=root, 
      train=True, 
      transform=rg_config['train_transform'],
      generate=generate,
      dataset_size=dataset_size,
      test_size=test_size,
      img_size=img_size,
      object_size=object_size,
      nb_objects=nb_objects,
      test_id_analogy=test_id_analogy,
      test_id_analogy_threshold=test_id_analogy_threshold)
    
    test_dataset = ReferentialGym.datasets.SortOfCLEVRDataset(root=root, 
      train=False, 
      transform=rg_config['test_transform'],
      generate=False,
      dataset_size=dataset_size,
      test_size=test_size,
      img_size=img_size,
      object_size=object_size,
      nb_objects=nb_objects,
      test_id_analogy=test_id_analogy,
      test_id_analogy_threshold=test_id_analogy_threshold)

  
  
  ## Modules:
  modules = {}

  from ReferentialGym import modules as rg_modules

  # MHCM:
  if 'Sort-of-CLEVR' in args.dataset:
    if 'Santoro2017' in args.arch:
      # Baseline:
      baseline_vm_id = f"baseline_{agent_config['architecture']}"
      baseline_vm_config = copy.deepcopy(agent_config)
      obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
      baseline_vm_config['obs_shape'] = obs_shape
      baselien_vm_input_stream_ids = {
        "losses_dict":"losses_dict",
        "logs_dict":"logs_dict",
        "signals:mode":"mode",
        "current_dataloader:sample:speaker_experiences":"inputs",
      }


      fm_id = "flatten0"
      fm_input_stream_keys = [
        f"modules:{baseline_vm_id}:ref:encoder:features",
      ]
    
      rrm_id = "reshaperepeat0"
      rrm_config = {
        'new_shape': [(1,-1)],
        'repetition': [(nb_questions,1)]
      }
      rrm_input_stream_keys = [
        "modules:flatten0:output_0",  # Baseline
      ]

      sqm_id = "squeeze_qas"
      sqm_config = {
        'dim': [None],
        #'inplace': True,
      }

      sqm_input_stream_keys = []
      for r_subtype_id in range(nb_r_qs):
        sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_relational_questions_{r_subtype_id}")
        sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_relational_answers_{r_subtype_id}")
        
      for nr_subtype_id in range(nb_nr_qs):
        sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_non_relational_questions_{nr_subtype_id}")
        sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_non_relational_answers_{nr_subtype_id}")

      cm_r_id = {}
      cm_r_config = {}
      cm_r_input_stream_keys = {}

      cm_nr_id = {}
      cm_nr_config = {}
      cm_nr_input_stream_keys = {}

      mhcm_r_id = {}
      mhcm_r_config = {}
      mhcm_r_input_stream_ids = {}

      mhcm_nr_id = {}
      mhcm_nr_config = {}
      mhcm_nr_input_stream_ids = {}

      # Baseline:
      b_cm_r_id = {}
      b_cm_r_config = {}
      b_cm_r_input_stream_keys = {}

      b_cm_nr_id = {}
      b_cm_nr_config = {}
      b_cm_nr_input_stream_keys = {}

      b_mhcm_r_id = {}
      b_mhcm_r_config = {}
      b_mhcm_r_input_stream_ids = {}

      b_mhcm_nr_id = {}
      b_mhcm_nr_config = {}
      b_mhcm_nr_input_stream_ids = {}
      
      feature_size = 4111
      mhcm_heads_arch = [2000,2000,2000,2000, 2000,1000,500,100]
      if args.resizeDim == 75 and 'Santoro2017-SoC-CNN' in args.arch:
        feature_size = 4111

      if args.resizeDim == 75 and 'Santoro2017-CLEVR-CNN' in args.arch:
        feature_size = 399
        #mhcm_heads_arch = [256,256,256,256, 256,'256-DP0.5',]
        mhcm_heads_arch = [256,'256-DP0.5',]
      
      mhcm_input_shape = feature_size

      for subtype_id in range(max(nb_r_qs,nb_nr_qs)):
        # Baseline:
        if subtype_id < nb_r_qs:
          b_cm_r_id[subtype_id] = f"baseline_concat_relational_{subtype_id}"
          b_cm_r_config[subtype_id] = {
            'dim': -1,
          }
          b_cm_r_input_stream_keys[subtype_id] = [
            "modules:reshaperepeat0:output_0",  # baseline visual features
            f"modules:squeeze_qas:output_{2*subtype_id}", #0~2*(nb_r_qs-1):2 (answers are interweaved...)
          ]

          b_mhcm_r_id[subtype_id] = f"baseline_mhcm_relational_{subtype_id}"
          b_mhcm_r_config[subtype_id] = {
            'loss_id': b_mhcm_r_id[subtype_id],
            'heads_output_sizes':[n_answers],
            'heads_archs':[
              mhcm_heads_arch,
            ],
            'input_shape': mhcm_input_shape,
            'detach_input': False,
            "use_cuda":args.use_cuda,
          }
          b_mhcm_r_input_stream_ids[subtype_id] = {
            f"modules:baseline_concat_relational_{subtype_id}:output_0":"inputs",
            f"modules:squeeze_qas:output_{2*subtype_id+1}":"targets", #1~2*nb_r_qs-1:2 (questions are interweaved...)
            "losses_dict":"losses_dict",
            "logs_dict":"logs_dict",
            "signals:mode":"mode",
          }

        if subtype_id < nb_nr_qs:
          b_cm_nr_id[subtype_id] = f"baseline_concat_non_relational_{subtype_id}"
          b_cm_nr_config[subtype_id] = {
            'dim': -1,
          }
          b_cm_nr_input_stream_keys[subtype_id] = [
            "modules:reshaperepeat0:output_0",  # baseline visual features
            f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id}", #2*nb_r_qs~2*nb_r_qs+2*(nb_nr_qs-1):2 (answers are interweaved...)
          ]

          b_mhcm_nr_id[subtype_id] = f"baseline_mhcm_non_relational_{subtype_id}"
          b_mhcm_nr_config[subtype_id] = {
            'loss_id': b_mhcm_nr_id[subtype_id],
            'heads_output_sizes':[n_answers],
            'heads_archs':[
              mhcm_heads_arch,
            ],
            'input_shape': mhcm_input_shape,
            'detach_input': False,
            "use_cuda":args.use_cuda,
          }
          b_mhcm_nr_input_stream_ids[subtype_id] = {
            f"modules:baseline_concat_non_relational_{subtype_id}:output_0":"inputs",
            f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id+1}":"targets", #2*nb_r_qs+1~2*nb_r_qs+2*nb_nr_qs-1:2 (answers are interweaved...)
            "losses_dict":"losses_dict",
            "logs_dict":"logs_dict",
            "signals:mode":"mode",
          }
    elif 'An2018-CNN' in args.arch:
      raise NotImplementedError

  # Building modules:

  if 'Sort-of-CLEVR' in args.dataset:
    #Baseline :
    modules[baseline_vm_id] = rg_modules.build_VisualModule(
      id=baseline_vm_id, 
      config=baseline_vm_config,
      input_stream_ids=baselien_vm_input_stream_ids)

    modules[fm_id] = rg_modules.build_FlattenModule(
      id=fm_id,
      input_stream_keys=fm_input_stream_keys)
    modules[rrm_id] = rg_modules.build_BatchReshapeRepeatModule(
      id=rrm_id,
      config=rrm_config,
      input_stream_keys=rrm_input_stream_keys)
    modules[sqm_id] = rg_modules.build_SqueezeModule(
      id=sqm_id,
      config=sqm_config,
      input_stream_keys=sqm_input_stream_keys)

    # Baseline:
    for subtype_id in range(max(nb_nr_qs,nb_r_qs)):
      if subtype_id < nb_r_qs:
        modules[b_cm_r_id[subtype_id]] = rg_modules.build_ConcatModule(
          id=b_cm_r_id[subtype_id],
          config=b_cm_r_config[subtype_id],
          input_stream_keys=b_cm_r_input_stream_keys[subtype_id])
        modules[b_mhcm_r_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
          id=b_mhcm_r_id[subtype_id], 
          config=b_mhcm_r_config[subtype_id],
          input_stream_ids=b_mhcm_r_input_stream_ids[subtype_id])
      if subtype_id < nb_nr_qs:
        modules[b_cm_nr_id[subtype_id]] = rg_modules.build_ConcatModule(
          id=b_cm_nr_id[subtype_id],
          config=b_cm_nr_config[subtype_id],
          input_stream_keys=b_cm_nr_input_stream_keys[subtype_id])
        modules[b_mhcm_nr_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
          id=b_mhcm_nr_id[subtype_id], 
          config=b_mhcm_nr_config[subtype_id],
          input_stream_ids=b_mhcm_nr_input_stream_ids[subtype_id])
  else:
    raise NotImplementedError

  homo_id = "homo0"
  homo_config = {"use_cuda":args.use_cuda}
  if args.homoscedastic_multitasks_loss:
    modules[homo_id] = rg_modules.build_HomoscedasticMultiTasksLossModule(
      id=homo_id,
      config=homo_config,
    )
  
  ## Pipelines:
  pipelines = {}

  # 0) Now that all the modules are known, let us build the optimization module:
  optim_id = "global_optim"
  optim_config = {
    "modules":modules,
    "learning_rate":args.lr,
    "with_gradient_clip":rg_config["with_gradient_clip"],
    "adam_eps":rg_config["adam_eps"],
  }

  optim_module = rg_modules.build_OptimizationModule(
    id=optim_id,
    config=optim_config,
  )
  modules[optim_id] = optim_module
  
  if 'Sort-of-CLEVR' in args.dataset:
    # Baseline:
    pipelines[baseline_vm_id] =[
      baseline_vm_id
    ]

    # Flatten and Reshape+Repeat:
    pipelines[rrm_id+"+"+sqm_id] = [
      fm_id,
      rrm_id,
      sqm_id
    ]

    # Compute relational items:
    for subtype_id in range(max(nb_r_qs,nb_nr_qs)):
      if subtype_id < nb_r_qs:
        #Baseline:
        pipelines[b_mhcm_r_id[subtype_id]] = [
          b_cm_r_id[subtype_id],
          b_mhcm_r_id[subtype_id]
        ]

      if subtype_id < nb_nr_qs:
        #Baseline:
        pipelines[b_mhcm_nr_id[subtype_id]] = [
          b_cm_nr_id[subtype_id],
          b_mhcm_nr_id[subtype_id]
        ]
  
  pipelines[optim_id] = []
  if args.homoscedastic_multitasks_loss:
    pipelines[optim_id].append(homo_id)
  pipelines[optim_id].append(optim_id)

  rg_config["modules"] = modules
  rg_config["pipelines"] = pipelines


  dataset_args = {
      "dataset_class":            "DualLabeledDataset",
      "modes": {"train": train_dataset,
                "test": test_dataset,
                },
      "need_dict_wrapping":       need_dict_wrapping,
      "nbr_stimulus":             rg_config['nbr_stimulus'],
      "distractor_sampling":      rg_config['distractor_sampling'],
      "nbr_distractors":          rg_config['nbr_distractors'],
      "observability":            rg_config['observability'],
      "object_centric":           rg_config['object_centric'],
      "descriptive":              rg_config['descriptive'],
      "descriptive_target_ratio": rg_config['descriptive_target_ratio'],
  }

  refgame = ReferentialGym.make(config=rg_config, dataset_args=dataset_args)

  # In[22]:

  refgame.train(nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

  logger.flush()

if __name__ == '__main__':
    main()
