import logging
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

class TrainingLogger:
    def __init__(self, log_dir, use_tensorboard=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('mamba_rtdetr')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.writer = None
        if use_tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir)
            self.info(f"TensorBoard logging enabled in {log_dir}")
        elif use_tensorboard:
            self.info("TensorBoard not found. pip install tensorboard to enable.")

    def log_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_value_dict, step):
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)

    def log_metrics(self, metrics_dict, step, prefix='train'):
        for k, v in metrics_dict.items():
            self.log_scalar(f'{prefix}/{k}', v, step)

    def log_lr(self, lr, step):
        self.log_scalar('train/lr', lr, step)

    def log_model_info(self, model):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.info(f"Model parameters: {num_params / 1e6:.2f} M")

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        if self.writer:
            self.writer.close()
