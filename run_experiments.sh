#!/bin/bash

# nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_nohyp_x_percent_5p > /dev/null 2>&1 && \
# nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_nohyp_x_percent_10p > /dev/null 2>&1 && \
# nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_nohyp_x_percent_20p > /dev/null 2>&1 && \
# nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_nohyp_x_percent_30p > /dev/null 2>&1 && \
nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_x_percent_5p > /dev/null 2>&1 && \
nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_x_percent_10p > /dev/null 2>&1 && \
nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_x_percent_20p > /dev/null 2>&1 && \
nohup python floppy/run_flop_counter.py --experiment=dist_agent_1lm_randrot_x_percent_30p > /dev/null 2>&1 &