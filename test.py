import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from dataset.dataloader import getloader
from dataset.bc_dataloader import getloader_bc
from model.mbc import MultiModalbileductClassifier
from model.ibc import ImageBileductClassifier
from utils.loss import get_optimizer_loss_scheduler
from utils.util import logdir, get_model_parameters

from monai.inferers import SlidingWindowInferer

from timeit import default_timer as timer
from tqdm import tqdm
import os
import wandb
import argparse
import warnings
warnings.filterwarnings('ignore')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

parser = argparse.ArgumentParser(description="Multimodal Bile duct stone Classfier")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--batch_size", default=100, type=int)
parser.add_argument("--num_gpus", default=8, type=int, help="Number of GPUs")
parser.add_argument("--num_classes", default=1, type=int, help="Assuming binary classification")
parser.add_argument("--optimizer", default='adam', type=str, help="Type of Optimizer") # 'adam', 'rmsprop'
parser.add_argument("--loss_function", default='BCE', type=str, help="Type of Loss function")
parser.add_argument("--scheduler", default='StepLR', type=str, help="Type of Learning rate scheduler") # 'stepLR','CosineAnnealingLR'
parser.add_argument("--momentum", default=0.0, type=float, help="Add momentum for SGD optimizer")
parser.add_argument("--model_architecture", default="efficientnet_b0", type=str, help="Model architecture")
parser.add_argument("--data_path", default='RawData/jpeg-melanoma-512x512/', type=str, help="Directory of data")
parser.add_argument("--log_dir", default='logs/', type=str, help="log directory")

args = parser.parse_args()
args.log_dir = logdir(args.log_dir)

PARAMS = {
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'num_gpus' : args.num_gpus, 
    'num_classes' : args.num_classes, 
    'optimizer': args.optimizer,
    'loss_function': args.loss_function, 
    'scheduler': args.scheduler, 
    'model_architecture': args.model_architecture,  
    'data_path' : args.data_path,
    'log_dir' : args.log_dir,
}
# PARAMS = get_model_parameters(PARAMS)

# wandb.init(project="Multimodal-Bileductstone-Classifier", save_code=True, name = PARAMS['model_architecture'], config=PARAMS)


    
# Training and Valaidation
class Tester:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, data_path, batch_size, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.data_path = data_path
        self.batch_size = batch_size
        self.log_dir = log_dir

    def test(self, epochs):
        self.model.eval()
        self.test_loss = 0.0
        self.correct = 0
        self.total = 0
        
        with torch.cuda.amp.autocast(): 
            test_loader = getloader_bc(self.data_path, batch_size=self.batch_size, mode='train')
            for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                image = batch["image"].to(device)
                output = window_infer(image, model)




# Create DataLoader and define model, optimizer, scheduler, loss_fn, and device
model = ImageBileductClassifier(PARAMS['num_classes'])
loss_fn, optimizer, scheduler = get_optimizer_loss_scheduler(PARAMS, model)
window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.25)
# Start time of training loop
start_training_time = timer()

trainer = Tester(model, optimizer, scheduler, loss_fn, device, PARAMS['data_path'], PARAMS['batch_size'], PARAMS['log_dir'])
train_losses, val_losses, train_accs, val_accs = trainer.train(PARAMS['epochs'], PARAMS['log_dir'])

# End time of training loop
end_training_time = timer()

# Calculate total training time
total_training_time = (end_training_time - start_training_time) / 3600
print(f"Total Training Time: {total_training_time:.2f} hours")