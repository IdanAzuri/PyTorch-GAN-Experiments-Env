# PyTorch-GAN-Experiments-Env
PyTorch GAN playground. 
Features:
1. yaml config file
2. Easy interface for Tensorboard
3. Generic system - just plug and play
 
 Implemented very simple versions of GAN and ACGAN

 ## Project Structure

    .
    ├── config                  # Config files (.yml, .json) using with hb-config
    ├── gan                     # Generative Adversarial Networks architecture 
        ├── __init__.py             # train, evaluate, predict logic
        ├── module.py               # Discriminator, Generator module
        └── utils.py                # Save and Load Model, TensorBoard
    ├── acgan                     # ACGAN architecture
        ├── __init__.py             # train,logic
        ├── module.py               # Discriminator, Generator module
        └── utils.py                # Save and Load Model, TensorBoard
    ├── data_loader.py          # make_data_loader (using DataLoader)
    ├── main.py                 
    ├── model.py                # define Model Spec
    └── utils.py                # utils
    └── plots.py                # generic functions to plot you results from a pickle file
    └── sampler.py              # different sampling methods:multi-modal uniform,muli-modal Gaussian,truncated normal...


Reference : [hb-config](https://github.com/hb-research/hb-config)

- Manage experiments like [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
## Create a new model
In order to implement your own model just create a **new branch**, copy the gan directory and edit it as you want.

## Data loader
Pytorch supports many datasets which are implemented to work with the data loaders, so if you want to work on different dataset just add it in the data_loader.py file and repsectively in the config file.
[Supported data sets](https://pytorch.org/docs/stable/torchvision/datasets.html)

## Config

Can control all **Experimental environment**.

example: config.yml

```yml
data:
  path: "data/"

model:
  z_dim: 100     # Random noise dimension coming into generator, per output vector
  real_dim: 784

  g_h1: 256
  g_h2: 512
  g_h3: 1024

  d_h1: 1024
  d_h2: 512
  d_h3: 256

  dropout: 0.3

train:
  model_dir: "logs/gan"
  batch_size: 64
  train_steps: 50000

  d_learning_rate: 0.0002  # 2e-4
  g_learning_rate: 0.0002
  optim_betas:
    - 0.9
    - 0.999

  save_checkpoints_steps: 1000
  verbose_step_count: 100

predict:
  batch_size: 64

slack:
  webhook_url: ""  # after training notify you using slack-webhook
```


## Usage

Install requirements.

```pip install -r requirements.txt```

Then, start training

```python main.py --mode train```

After training, generate images

```python main.py --mode predict```






<br><br><br>



###### Some code adapted from [Dongjun Lee](https://github.com/DongjunLee)
