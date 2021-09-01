import torch as th


def save_checkpoint(net, epoch, loss, optimizer, path, hparams):
    
    th.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'hparams' : hparams
                }, path)
    

def load_checkpoint()