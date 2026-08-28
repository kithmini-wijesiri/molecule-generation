import os
import torch
import random
import numpy as np
import torch.distributed as dist
from torch.autograd import Variable


def init_distributed():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return 0, 0, 1, False

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def get_model_module(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def reduce_value(value, device, world_size):
    if world_size == 1:
        return value

    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def save_checkpoint(epoch, model, optimizer, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint = dict(
        epoch=epoch,
        model=get_model_module(model).state_dict(),
        optimizer=optimizer.state_dict()
    )
    
    print("Saving epoch checkpoint...")
    torch.save(checkpoint, os.path.join(weights_dir, f'{epoch}.pt'))
    print(f"Saved {weights_dir}/{epoch}.pt")

    print("Saving best checkpoint...")
    torch.save(checkpoint, os.path.join(weights_dir, 'best.pt'))
    print(f"Saved {weights_dir}/best.pt")

def set_randomness(random_seed: int = 2022):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if opt.use_cond2dec == True:
        cond_mask = np.zeros((1, opt.cond_dim, opt.cond_dim))
        cond_mask_upperright = np.ones((1, opt.cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, opt.cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    return np_mask.to(opt.device)


def create_masks(src, trg, cond, opt):
    torch.set_printoptions(profile="full")
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    cond_mask = torch.unsqueeze(cond, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    src_mask = torch.cat([cond_mask, src_mask], dim=2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        if opt.use_cond2dec == True:
            trg_mask = torch.cat([cond_mask, trg_mask], dim=2)
        np_mask = nopeak_mask(trg.size(1), opt)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None

    return src_mask, trg_mask
