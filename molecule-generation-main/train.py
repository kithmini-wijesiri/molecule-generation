from pickle import dump
import os
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler

from args import get_args, get_weights_dir, save_experiment_config
from data_loader import *
from losses import loss_function, KLAnnealer
from models.transformer import Transformer
from utils import (
    set_randomness,
    create_masks,
    save_checkpoint,
    init_distributed,
    cleanup_distributed,
    is_main_process,
    reduce_value,
)


def train(opt, model, optimizer, scheduler, train_loader, valid_loader, rank, world_size, weights_dir):

    best_loss = np.inf
    for epoch in range(opt.epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if opt.use_KLA:
            beta = min(
                KLAnnealer(opt, epoch),
                opt.KLA_max_beta
            )
        else:
            beta = 1.0
        if is_main_process(rank):
            print(f"Epoch {epoch}: KL beta = {beta:.2f}")

        total_loss = 0
        RCE_mol_loss = 0
        RCE_prop_loss = 0
        KLD_loss = 0

        model.train()
        train_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            ncols=100,
            disable=not is_main_process(rank),
        )
        for i, (src, trg, cond) in enumerate(train_pbar):
            src = src.transpose(0, 1).to(opt.device)
            trg = trg.transpose(0, 1).to(opt.device)
            cond = cond.float().to(opt.device)

            trg_input = trg[:, :-1]

            src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
            preds_prop, preds_mol, mu, log_var, z = model(
                src, trg_input, cond, src_mask, trg_mask)

            preds_mol = preds_mol.contiguous().view(-1, preds_mol.size(-1))
            preds_prop = preds_prop.view(-1, 3)
            ys_mol = trg[:, 1:].contiguous().view(-1)
            cond = cond.view(-1, 3)

            loss, RCE_mol, RCE_prop, KLD = loss_function(
                opt, beta, preds_prop, preds_mol, cond, ys_mol, mu, log_var
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += (loss.item() / opt.batch_size)
            RCE_mol_loss += (RCE_mol.item() / opt.batch_size)
            RCE_prop_loss += (RCE_prop.item() / opt.batch_size)
            KLD_loss += (KLD.item() / opt.batch_size)
            train_pbar.set_postfix({
                'total': total_loss / (i + 1),
                'RCE_mol': RCE_mol_loss / (i + 1),
                'RCE_prop': RCE_prop_loss / (i + 1),
                'KLD': KLD_loss / (i + 1),
                'beta': beta
            })

        total_loss = reduce_value(total_loss, opt.device, world_size)
        RCE_mol_loss = reduce_value(RCE_mol_loss, opt.device, world_size)
        RCE_prop_loss = reduce_value(RCE_prop_loss, opt.device, world_size)
        KLD_loss = reduce_value(KLD_loss, opt.device, world_size)

        if is_main_process(rank):
            print(
                f'train {epoch} | {total_loss} {RCE_mol_loss} {RCE_prop_loss} {KLD_loss}')

        model.eval()
        valid_loss, valid_RCE_mol_loss, valid_RCE_prop_loss, valid_KLD_loss = 0, 0, 0, 0
        valid_pbar = tqdm(
            valid_loader,
            total=len(valid_loader),
            ncols=100,
            disable=not is_main_process(rank),
        )
        with torch.no_grad():
            for i, (src, trg, cond) in enumerate(valid_pbar):
                src = src.transpose(0, 1).to(opt.device)
                trg = trg.transpose(0, 1).to(opt.device)
                cond = cond.float().to(opt.device)

                trg_input = trg[:, :-1]

                src_mask, trg_mask = create_masks(
                    src, trg_input, cond, opt
                )
                preds_prop, preds_mol, mu, log_var, z = model(
                    src, trg_input, cond, src_mask, trg_mask
                )
                ys_mol = trg[:, 1:].contiguous().view(-1)

                preds_mol = preds_mol.contiguous().view(-1, preds_mol.size(-1))
                ys_mol = trg[:, 1:].contiguous().view(-1)
                preds_prop = preds_prop.view(-1, 3)
                cond = cond.view(-1, 3)

                loss_te, RCE_mol_te, RCE_prop_te, KLD_te = loss_function(
                    opt, beta, preds_prop, preds_mol, cond, ys_mol, mu, log_var
                )
                valid_loss += (loss_te.item() / opt.batch_size)
                valid_RCE_mol_loss += (RCE_mol_te.item() / opt.batch_size)
                valid_RCE_prop_loss += (RCE_prop_te.item() / opt.batch_size)
                valid_KLD_loss += (KLD_te.item() / opt.batch_size)
                valid_pbar.set_postfix({
                    'total': valid_loss,
                    'RCE_mol': valid_RCE_mol_loss,
                    'RCE_prop': valid_RCE_prop_loss,
                    'KLD': valid_KLD_loss
                })

        valid_loss = reduce_value(valid_loss, opt.device, world_size)
        valid_RCE_mol_loss = reduce_value(valid_RCE_mol_loss, opt.device, world_size)
        valid_RCE_prop_loss = reduce_value(valid_RCE_prop_loss, opt.device, world_size)
        valid_KLD_loss = reduce_value(valid_KLD_loss, opt.device, world_size)

        scheduler.step()
        if is_main_process(rank):
            print(
                f'valid {epoch} | {valid_loss} {valid_RCE_mol_loss} {valid_RCE_prop_loss} {valid_KLD_loss}')

        if best_loss > valid_loss:
            best_loss = valid_loss
            if is_main_process(rank):
                save_checkpoint(epoch, model, optimizer, weights_dir)
                print('best update !!! best loss: ', best_loss)


def get_optimizer(model, config):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 betas=(.9, .98))

    return optimizer


def get_scheduler(optimizer, config):

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(10, config.epochs, 10)),
        gamma=0.95,  # config.lr_gamma,
        last_epoch=-1,
    )

    return lr_scheduler


if __name__ == "__main__":
    rank, local_rank, world_size, is_distributed = init_distributed()

    try:
        set_randomness()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        opt = get_args()
        weights_dir = get_weights_dir(opt)
        if is_main_process(rank):
            print(f'Using experiment weights dir: {weights_dir}')
        if is_distributed:
            opt.device = torch.device(f'cuda:{local_rank}')
        else:
            opt.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

        MAX_LEN = opt.max_len

        if is_main_process(rank):
            df = pd.read_csv('smiles.csv')
            train_data, valid_data = train_test_split(
                df, test_size=0.2, random_state=opt.random_seed
            )
            train_data = train_data[train_data['smiles'].str.len() <= MAX_LEN].reset_index(drop=True)
            valid_data = valid_data[valid_data['smiles'].str.len() <= MAX_LEN].reset_index(drop=True)

            train_data, valid_data, scaler = property_normalize(train_data, valid_data)

            os.makedirs("weights", exist_ok=True)
            with open("weights/robust_scaler.pkl", "wb") as f:
                dump(scaler, f)

            print("Saved scaler: weights/robust_scaler.pkl")

            train_data.to_csv('train_prop.csv', index=False)
            valid_data.to_csv('valid_prop.csv', index=False)
            print('create train data and valid data...')

        if is_distributed:
            dist.barrier()

        train_data = pd.read_csv('train_prop.csv')
        valid_data = pd.read_csv('valid_prop.csv')

        train_dataset = MoleculeDataset(train_data, "smiles", "smiles")

        if is_main_process(rank):
            print(train_data.iloc[1], train_dataset[1])
            print(train_dataset.source_vocab.itos,
                  train_dataset.source_vocab.stoi)

        valid_dataset = MoleculeDataset(valid_data, "smiles", "smiles")

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        ) if is_distributed else None
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        ) if is_distributed else None

        train_loader = get_train_loader(
            train_dataset,
            batch_size=opt.batch_size,
            num_workers=2,
            sampler=train_sampler,
        )
        valid_loader = get_valid_loader(
            valid_dataset,
            train_dataset,
            batch_size=opt.batch_size,
            num_workers=2,
            sampler=valid_sampler,
        )

        model = Transformer(
            opt,
            len(train_dataset.source_vocab.itos),
            len(train_dataset.source_vocab.itos)
        )
        model.to(opt.device)

        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        optimizer = get_optimizer(model, opt)
        scheduler = get_scheduler(optimizer, opt)

        opt.src_pad = train_dataset.source_vocab.stoi['<PAD>']
        opt.trg_pad = train_dataset.source_vocab.stoi['<PAD>']
        if is_main_process(rank):
            save_experiment_config(opt, weights_dir)
            print('model training start...')
        train(
            opt,
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            rank,
            world_size,
            weights_dir,
        )
    finally:
        cleanup_distributed()
