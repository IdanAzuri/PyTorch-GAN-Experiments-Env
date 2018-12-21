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
    ├── data_loader.py          # make_data_loader (using DataLoader)
    ├── main.py                 
    ├── model.py                # define Model Spec
    └── model.py                # utils


Reference : [hb-config](https://github.com/hb-research/hb-config)

- Manage experiments like [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)


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

Adapted from [Dongjun Lee](https://github.com/DongjunLee)
