WANDB_CACHE_DIR=./wandb_cache/ xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m ipdb -c c benchmark_wandb_ether.py \
--seed=10 \
--project=ETHER \
--success_threshold=0.5 \
--use_cuda=True \
--config=room12x5Objs_miniworld_wandb_benchmark_OracleTHER+R2D2+RP+SharedObsEncoder_config.yaml \
--language_guided_curiosity=False \
--coverage_manipulation_metric=True \
--MiniWorld_entity_visibility_oracle=False \
--MiniWorld_entity_visibility_oracle_top_view=False \
--use_ETHER=False --use_THER=True \
--use_RP=False --RP_use_RP=True \
--use_ELA=False --ELA_use_ELA=True \
--use_HER=False --goal_oriented=False \
--ETHER_use_ETHER=False \
--THER_use_THER=True \
--THER_use_THER_predictor_supervised_training=False \
--THER_use_THER_predictor_supervised_training_data_collection=True \
--semantic_embedding_init='none' \
--semantic_prior_mixing='multiplicative' \
--semantic_prior_mixing_with_detach=False \
--RP_use_PER=True \
--RP_lock_test_storage=False \
--RP_predictor_learning_rate=6.25e-5 \
--RP_gradient_clip=5.0 \
--RP_replay_capacity=16384 --RP_min_capacity=32 \
--RP_predictor_nbr_minibatches=4 --RP_predictor_batch_size=256 \
--RP_predictor_test_train_split_interval=3 --RP_test_replay_capacity=1024 \
--RP_test_min_capacity=32 --RP_replay_period=1024 \
--RP_nbr_training_iteration_per_update=8 \
--RP_predictor_accuracy_threshold=90 \
--THER_use_PER=True --THER_observe_achieved_goal=True \
--THER_lock_test_storage=True \
--THER_feedbacks_failure_reward=-1 --THER_feedbacks_success_reward=1 \
--THER_episode_length_reward_shaping=True \
--THER_replay_capacity=2 --THER_min_capacity=8 \
--THER_predictor_nbr_minibatches=1 --THER_predictor_batch_size=32 \
--THER_predictor_test_train_split_interval=5 --THER_test_replay_capacity=2 \
--THER_test_min_capacity=4 --THER_replay_period=4096 \
--THER_train_on_success=False --THER_nbr_training_iteration_per_update=0 \
--THER_predict_PADs=False --THER_predictor_accuracy_threshold=0.95 \
--THER_predictor_accuracy_safe_to_relabel_threshold=0.0 --THER_filter_predicate_fn=True \
--THER_relabel_terminal=False --THER_filter_out_timed_out_episode=False \
--THER_train_contrastively=False --THER_contrastive_training_nbr_neg_examples=0 \
--THER_timing_out_episode_length_threshold=40 \
--BabyAI_Bot_action_override=False \
--n_step=3 --nbr_actor=32 --eps_greedy_alpha=2.0 \
--nbr_minibatches=1 --batch_size=64 \
--min_capacity=1e3 --replay_capacity=5e3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.5 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False --HER_target_clamping=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--nbr_training_iteration_per_cycle=1 --nbr_episode_per_cycle=0 \
--single_pick_episode=True \
--time_limit=40 \
--train_observation_budget=1.0e6

#--train_observation_budget=300000 
#WARNING : as --THER_observe_achieved_goal=True, the BehaviourDescriptionWrapper is operational.
# It means that there is no need to use --MiniWorld_entity_visibility_oracle=True.
# But, EoS is the default description when nothing is carried.
# TODO: Using --THER_observe_all_description=True would enable the BehaviourDescriptionWrapper to describe all states with a list of object that are visible, ordered by proximity.


