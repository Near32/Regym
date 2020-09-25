python -m cProfile -o r2d2.pstat benchmark_atari.py pong_1M_benchmark_r2d2.yaml
pyprof2calltree -i r2d2.pstat -o r2d2.calltree
kcachegrind r2d2.calltree

#https://stackoverflow.com/a/37157132