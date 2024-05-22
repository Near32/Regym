# There are 3 to 4 import_ipdb that will pop out just to sanity check that the argument are taken into account.
# Just enter 'c' to continue...
# If there is a break after the environments have been created, then it is an actual issue (unless have forgotten to remove a debugging set_trace(), sorry...)

# With communicating rule-based agent:
python -m ipdb -c c ./benchmark_selfplay_comaze.py comaze_communicating_rule_based_benchmark_config.yaml --pubsub --communicating_rule_based --use_ms_cic

# With action-only rule-based agent:
#python -m ipdb -c c ./benchmark_selfplay_comaze.py comaze_communicating_rule_based_benchmark_config.yaml --pubsub --rule_based --use_ms_cic
