import torch
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from beam_search import beam_search
from pickle import load
from sklearn.model_selection import train_test_split

from args import get_args, get_weights_dir, load_experiment_config, apply_experiment_config
from data_loader import *
from utils import set_randomness
from models.transformer import Transformer


def get_sampled_element(myCDF):
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF >= a)-1


def run_sampling(xc, dxc, myPDF, myCDF, nRuns):
    sample_list = []
    X = np.zeros_like(myPDF, dtype=int)
    for k in np.arange(nRuns):
        idx = get_sampled_element(myCDF)
        sample_list.append(xc[idx] + dxc * np.random.normal() / 2)
        X[idx] += 1
    return np.array(sample_list).reshape(nRuns, 1), X/np.sum(X)


def tokenlen_gen_from_data_distribution(data, nBins, size):
    count_c, bins_c, = np.histogram(data, bins=nBins)
    myPDF = count_c / np.sum(count_c)
    dxc = np.diff(bins_c)[0]
    xc = bins_c[0:-1] + 0.5 * dxc

    myCDF = np.zeros_like(bins_c)
    myCDF[1:] = np.cumsum(myPDF)

    tokenlen_list, X = run_sampling(xc, dxc, myPDF, myCDF, size)

    return tokenlen_list


def inference(
    opt,
    scaler,
    model,
    SRC,
    TRG,
    n_samples: int = 1,
    n_per_samples: int = 100
):
    molecules = []
    valid_check = []
    conds_rdkit = []
    conds_trg = []

    data = pd.read_csv("valid_prop.csv")
    data['length'] = data['smiles'].apply(lambda x: len(str(x)))
    toklen_data = data['length'].values
    print('toklen_data.max(), toklen_data.min(): ',
          toklen_data.max(), toklen_data.min())
    conds = data[['weight', 'logp', 'TPSA']][:n_per_samples].values
    toklen_data = tokenlen_gen_from_data_distribution(
        data=toklen_data,
        nBins=int(toklen_data.max()-toklen_data.min()),
        size=n_samples*n_per_samples
    )

    for idx in tqdm(range(n_per_samples), ncols=100):
        for i in range(n_samples):
            # +3 due to cond2enc
            #toklen = int(toklen_data[idx]) + 3
            toklen = int(toklen_data[idx].item()) + 3
            z = torch.Tensor(np.random.normal(
                size=(1, toklen, opt.latent_dim)))

            cond = torch.autograd.Variable(torch.Tensor(conds[idx]))
            gen_mol = beam_search(cond, model, SRC, TRG, toklen, opt, z)
            gen_mol = ''.join(gen_mol).replace(' ', '')
            molecules.append(gen_mol)
            conds_trg.append(
                scaler.inverse_transform(conds[[idx]]).reshape(1, 3)[0]
            )

            m = Chem.MolFromSmiles(gen_mol)
            if m is None:
                valid_check.append(0)
                conds_rdkit.append([None, None, None, None])
            else:
                valid_check.append(1)
                conds_rdkit.append(np.array([Descriptors.MolWt(m),
                                            Descriptors.MolLogP(m),
                                            Descriptors.TPSA(m),
                                            QED.qed(m)]))

    np_conds_trg = np.array(conds_trg)
    np_conds_rdkit = np.array(conds_rdkit)
    print(np_conds_trg.shape)
    gen_list = pd.DataFrame({
        "mol": molecules,
        "validity": valid_check,
        "condition(weight)": np_conds_trg[:, 0],
        "condition(logP)": np_conds_trg[:, 1],
        "condition(TPSA)": np_conds_trg[:, 2],
        "rdkit(weight)": np_conds_rdkit[:, 0],
        "rdkit(logP)": np_conds_rdkit[:, 1],
        "rdkit(TPSA)": np_conds_rdkit[:, 2],
        "rdkit(QED)": np_conds_rdkit[:, 3],
    })

    gen_list.to_csv(
        f'results/generation_results_{opt.experiment_name}_{n_samples}.csv',
        index=True,
    )

    print(f'generation success ratio: {sum(valid_check)/len(valid_check)}')


if __name__ == "__main__":
    set_randomness()

    opt = get_args()
    weights_dir = get_weights_dir(opt)
    config = load_experiment_config(weights_dir)
    if config:
        opt = apply_experiment_config(opt, config)
        print(f'Loaded experiment config from {weights_dir}/config.json')
    else:
        print(
            f'Warning: no config.json in {weights_dir}; '
            'using CLI flags for model architecture.'
        )

    opt.experiment_name = os.path.basename(weights_dir)
    save_path = os.path.join(weights_dir, 'best.pt')
    print(f'Using checkpoint: {save_path}')

    train_data = pd.read_csv("train_prop.csv")

    train_dataset = MoleculeDataset(
        train_data,
        "smiles",
        "smiles"
    )

    opt.src_pad = train_dataset.source_vocab.stoi['<PAD>']
    opt.trg_pad = train_dataset.source_vocab.stoi['<PAD>']

    SRC = train_dataset.source_vocab.stoi
    TRG = train_dataset.source_vocab.itos
    
    # Create model
    model = Transformer(
        opt,
        len(train_dataset.source_vocab.stoi),
        len(train_dataset.source_vocab.itos)
    )
    
    # Load checkpoint
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model"])
    
    # Move to GPU and evaluation mode
    model.to(opt.device)
    model.eval()
   
    # Load scaler
    robust_scaler = load(open('weights/robust_scaler.pkl', 'rb'))
    
    # Generate molecules
    inference(opt, robust_scaler, model, SRC, TRG)
