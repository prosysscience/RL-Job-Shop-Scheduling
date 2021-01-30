A Good Environment Is All You Need
==============================

This folder contains the implementation of the paper "A Good Environment Is All You Need".

It contains the deep reinforcement learning approach we have developed to solve the Job-Shop Scheduling problem.

The optimized environment is available as a separate [repository](https://github.com/prosysscience/JSSEnv).

![til](./ta01.gif)

Getting Started
------------

This work uses Ray's RLLib, Tensorflow and Wandb.

Make sure you `git`, `cmake`, `zlib1g`, and, on Linux, `zlib1g-dev` installed.

You need also to have a Weight and Bias account to log your metrics. 
Otherwise, just remove all occurrence of wandb and log the metrics another way.

```shell
git clone LINK_TO_REPOSITORY
cd JSS
pip install -r requirements.txt
```

### Important: Your instance must follow [Taillard's specification](http://jobshop.jjvh.nl/explanation.php#taillard_def). 

Project Organization
------------

    ├── README.md                 <- The top-level README for developers using this project.
    └── JSS
        |── dispatching_rules      <- Contains the code to run the disptaching rule FIFO and MWTR
        |── randomLoop             <- A random loop with action mask, usefull to debug environment and
        |                             to check if our agent learn
        └── instances              <- All Taillard's instances + 5 Demirkol instances

--------

## License

MIT License