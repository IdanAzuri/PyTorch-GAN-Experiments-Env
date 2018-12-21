# PyTorch-deep-photo-styletransfer
PyTorch GAN playground. 
Features:
1. yaml config file
2. Easy interface for Tensorboard
3. Generic system - just plug and play
 
 Implemented very simple versions of GAN and ACGAN

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

- generated image example
