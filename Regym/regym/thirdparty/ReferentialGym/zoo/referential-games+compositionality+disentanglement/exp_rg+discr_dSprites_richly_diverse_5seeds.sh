# $1==SEED_BASIS  
# $2==MAX_SENTENCE_LENGTH (10/20)
# $3==VOCAB_SIZE1 (10,100)
# $4==BATCH_SIZE (2/12/24/36/48)
# $5==VAE_GAMMA_FACTOR (0.0--100.0)
# $6==NBR_DISTRACTORRS (0--)
# $7==GRAPHTYPE ("straight_through_gumbel_softmax"/"obverter")
# $8==DESCRIPTIVE_RATIO (0.0--)
# $9==OBJECT_CENTRIC (""/"--object_centric")
# $10==SHARED_ARCH (""/"--shared_architecture")
# $11==SAMPLING/BASELINE (""/"--baseline_only"/"--obverter_sampling_round_alternation_only")

#python -m ipdb -c c RG/zoo/referential-games+compositionality+disentanglement/train.py \
python -m ipdb -c c train.py \
--parent_folder ./RichlyDiverseStimuli \
--use_cuda --seed $(($1+0)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.75 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type NLL --graphtype $7 \
--metric_epoch_period 200 --nbr_train_points 4000 --nbr_eval_points 2000 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive --descriptive_ratio $8 \
$9 ${10} ${11} \
--nb_3dshapespybullet_shapes 10 --nb_3dshapespybullet_colors 10 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-5-S3-X-5-S3-Orientation-4-N-Scale-1-S3-Shape-1-N &
#--vae_factor_gamma $5 \
#--with_baseline \

python -m ipdb -c c train.py \
--parent_folder ./RichlyDiverseStimuli \
--use_cuda --seed $(($1+10)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.75 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type NLL --graphtype $7 \
--metric_epoch_period 200 --nbr_train_points 4000 --nbr_eval_points 2000 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive --descriptive_ratio $8 \
$9 ${10} ${11} \
--nb_3dshapespybullet_shapes 10 --nb_3dshapespybullet_colors 10 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-5-S3-X-5-S3-Orientation-4-N-Scale-1-S3-Shape-1-N &
#--vae_factor_gamma $5 \
#--with_baseline \

python -m ipdb -c c train.py \
--parent_folder ./RichlyDiverseStimuli \
--use_cuda --seed $(($1+20)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.75 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type NLL --graphtype $7 \
--metric_epoch_period 200 --nbr_train_points 4000 --nbr_eval_points 2000 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive --descriptive_ratio $8 \
$9 ${10} ${11} \
--nb_3dshapespybullet_shapes 10 --nb_3dshapespybullet_colors 10 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-5-S3-X-5-S3-Orientation-4-N-Scale-1-S3-Shape-1-N &
#--vae_factor_gamma $5 \
#--with_baseline \

python -m ipdb -c c train.py \
--parent_folder ./RichlyDiverseStimuli \
--use_cuda --seed $(($1+30)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.75 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type NLL --graphtype $7 \
--metric_epoch_period 200 --nbr_train_points 4000 --nbr_eval_points 2000 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive --descriptive_ratio $8 \
$9 ${10} ${11} \
--nb_3dshapespybullet_shapes 10 --nb_3dshapespybullet_colors 10 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-5-S3-X-5-S3-Orientation-4-N-Scale-1-S3-Shape-1-N &
#--vae_factor_gamma $5 \
#--with_baseline \

python -m ipdb -c c train.py \
--parent_folder ./RichlyDiverseStimuli \
--use_cuda --seed $(($1+40)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.75 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type NLL --graphtype $7 \
--metric_epoch_period 200 --nbr_train_points 4000 --nbr_eval_points 2000 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive --descriptive_ratio $8 \
$9 ${10} ${11} \
--nb_3dshapespybullet_shapes 10 --nb_3dshapespybullet_colors 10 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-5-S3-X-5-S3-Orientation-4-N-Scale-1-S3-Shape-1-N
#--vae_factor_gamma $5 \
#--with_baseline \
