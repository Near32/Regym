python train_obverter.py \
--parent_folder ./Obverter/ObjectCentric/\
Tau1e0+DecisionHeads2+Normalize+InnerModelGen+OneMinusMaxProb+StopPadding+WholeSentence\
+StabEps1m8+ScaleNL0+AlwaysArgmax\
/SymbolEmb64+GRU64+CNN256+Decision128/ \
--use_cuda --seed $1 \
--obverter_nbr_games_per_round 20 --obverter_threshold_to_stop_message_generation 0.95 \
--emb_dropout_prob 0.0 --dropout_prob 0.0 --use_sentences_one_hot_vectors \
--batch_size 40 --mini_batch_size 256 --resizeDim 128 --arch BN+BaselineCNN \
--descriptive --descriptive_ratio $2 \
--max_sentence_length $3 --vocab_size $4 --epoch 4000 \
--symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--object_centric --nbr_train_distractors $5 --nbr_test_distractors $5 \
--obverter_use_decision_head \
--agent_loss_type NLL \
--metric_epoch_period 1 \
--nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 \
--nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 2 \
--lr 6e-4 

#--metric_fast
#--use_obverter_sampling \
#--nbr_train_distractors 0 --nbr_test_distractors 0 \
#--egocentric

#--emb_dropout_prob 0.5 --dropout_prob 0.0 --use_sentences_one_hot_vectors \

# --force_eos
#--resizeDim 32 --arch BN+3xCNN3x3
#--resizeDim 64 --arch BN+BaselineCNN
