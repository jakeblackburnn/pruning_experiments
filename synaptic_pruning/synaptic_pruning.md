Make an experiment replicating the synaptic pruning methodology described in the paper. These experiments should be comprized of python source files, and a culminating jupyter notebook going through and describing the experiments, plus a md report of results. 

the experiments themselves should:

1. replicate the results of the paper as close as can be done reasonably on this machine (m4 mac with metal/mlx gpu) using one of the datasets used in the paper downloaded from the internet 

2. do a 'bitter lesson' bullshit test, trying the same experiment at 3 levels of network size, 3 levels of compute scale and 3 levels of dataset size, (3x3x3 test grid) to observe how the pruning technique scales with compute and data. If the marginal benefit of the technique does not scale well with network size, flops, and data, our takeaway from this experiment may be 'bitter lesson pilled'.

if the experiments would be better run on a rtx5080, hand off to user to run experiments, rather than running then yourself.
