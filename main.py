import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
import numpy as np
from tqdm import tqdm
from utils import setup_logging, vis_density_GMM, vis_2D_samples, visualize_sampling
from gmm import GMM, GMMDataset
from grbm import GRBM
import wandb
import torch.nn.functional as F
from rbm import RBM, train_rbm

EPS = 1e-7
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)


def save(model, results_folder, epoch):
    data = {'epoch': epoch, 'model': model.state_dict()}
    torch.save(data, f'{results_folder}/model-{epoch}.pt')


def load(model, results_folder, epoch):
    data = torch.load(f'{results_folder}/model-{epoch}.pt')
    model.load_state_dict(data['model'])

class DataLoaderWrapper:
    def __init__(self, loader, transform=None, is_cuda=True):
        self.loader = loader
        self.transform = transform
        self.is_cuda = is_cuda

    def __iter__(self):
        for batch in self.loader:
            if self.transform:
                if self.is_cuda:
                    batch[0] = batch[0].cuda()
                shape = batch[0].shape
                batch[0] = self.transform(batch[0].view(shape[0], -1)).view(shape[0], -1)
            yield batch

    def __len__(self):
        return len(self.loader)

def train(model,
          train_loader,
          optimizer,
          config):
    model.train()
    for ii, (data, _) in enumerate(tqdm(train_loader)):
        if config['cuda']:
            data = data.cuda()

        optimizer.zero_grad()
        model.CD_grad(data)
        if config['clip_norm'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
        optimizer.step()

        if ii == len(train_loader) - 1:
            recon_loss = model.reconstruction(data).item()

    return recon_loss


def create_dataset(config):
    if 'GMM' in config['dataset']:
        if config['dataset'] == 'GMM_iso':
            # isotropic
            gmm_model = GMM(torch.tensor([0.33, 0.33, 0.34]),
                            torch.tensor([[-5, -5], [5, -5], [0, 5]]),
                            torch.tensor([[1, 1], [1, 1], [1, 1]])).cuda()
        else:
            # anisotropic
            gmm_model = GMM(torch.tensor([0.33, 0.33, 0.34]),
                            torch.tensor([[-5, -5], [5, -5], [0, 5]]),
                            torch.tensor([[1.25, 0.5], [1.25, 0.5], [0.5,
                                                                     1.25]])).cuda()

        vis_density_GMM(gmm_model, config)
        samples = gmm_model.sampling(config['num_samples'])
        vis_2D_samples(samples.cpu().numpy(), config, tags='ground_truth')
        train_set = GMMDataset(samples)
    elif config['dataset'] == 'MNIST':
        train_set = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(config['img_mean'],
                                                            config['img_std'])
                                   ]))
        test_set = datasets.MNIST('./data',
                                   train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(config['img_mean'],
                                                            config['img_std'])
                                   ]))
    elif config['dataset'] == 'CelebA':
        train_set = datasets.CelebA('./data',
                                    split='train',
                                    download=False,
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(
                                            config['crop_size']),
                                        transforms.Resize(config['height']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(config['img_mean'],
                                                             config['img_std'])
                                    ]))
        # for now due to limited compute coz of high K and deep architecture
        train_set = torch.utils.data.Subset(train_set, range(4000))
        test_set = datasets.CelebA('./data',
                                    split='test',
                                    download=False,
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(
                                            config['crop_size']),
                                        transforms.Resize(config['height']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(config['img_mean'],
                                                             config['img_std'])
                                    ]))
    elif config['dataset'] == 'CelebA2K':
        train_set = datasets.CelebA('./data',
                                    split='train',
                                    download=False,
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(
                                            config['crop_size']),
                                        transforms.Resize(config['height']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(config['img_mean'],
                                                             config['img_std'])
                                    ]))
        train_set = torch.utils.data.Subset(train_set, range(2000))
        test_set = datasets.CelebA('./data',
                                    split='test',
                                    download=False,
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(
                                            config['crop_size']),
                                        transforms.Resize(config['height']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(config['img_mean'],
                                                             config['img_std'])
                                    ]))
    elif config['dataset'] == 'FashionMNIST':
        train_set = datasets.FashionMNIST('./data',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(config['img_mean'],
                                                                   config['img_std'])
                                          ]))
        test_set = datasets.FashionMNIST('./data',
                                          train=False,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(config['img_mean'],
                                                                   config['img_std'])
                                          ]))
    if 'GMM' not in config['dataset']:
        config['img_mean'] = torch.tensor(config['img_mean'])
        config['img_std'] = torch.tensor(config['img_std'])

    return train_set, test_set


def train_model(args):
    """Let us train a GRBM and see how it performs"""
    pid = os.getpid()
    # Load config
    with open(f'config/{args.dataset}.json') as json_file:
        config = json.load(json_file)


    wandb_mode = None if args.use_wandb else 'disabled'
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="PGM",
        entity="vlr-3dmamba",
        # track hyperparameters and run metadata
        config=config,
        mode=wandb_mode,
    )

    # kludgey check for inpainting config
    if 'do_inpainting' not in config:
        config['do_inpainting'] = False
    if 'mask' not in config:
        config['mask'] = None
    if 'randomize_mask' not in config:
        config['randomize_mask'] = False
    if 'deep_hidden_sizes' not in config:
        config['deep_hidden_sizes'] = None
    if args.disable_deep:
        config['deep_hidden_sizes'] = None

    config['exp_folder'] = f"exp/inpainting={config['do_inpainting']}_{config['dataset']}_{config['model']}_{pid}_inference={config['inference_method']}_H={config['hidden_size']}_B={config['batch_size']}_CD={config['CD_step']}"
    
    if not os.path.isdir(config['exp_folder']):
        os.makedirs(config['exp_folder'])

    log_file = os.path.join(config['exp_folder'], f'log_exp_{pid}.txt')
    logger = setup_logging('INFO', log_file)
    logger.info('Writing log file to {}'.format(log_file))

    with open(os.path.join(config['exp_folder'], f'config_{pid}.json'),
              'w') as outfile:
        json.dump(config, outfile, indent=4)

    config['visible_size'] = config['height'] * \
        config['width'] * config['channel']
    train_set, test_set = create_dataset(config)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config['batch_size'],
                                               shuffle=True)
    if config['do_inpainting']: 
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=config['sampling_batch_size'],
                                                  shuffle=False)
    else:
        test_loader = None

    model = GRBM(config['visible_size'],
                 config['hidden_size'],
                 CD_step=config['CD_step'],
                 CD_burnin=config['CD_burnin'],
                 init_var=config['init_var'],
                 inference_method=config['inference_method'],
                 Langevin_step=config['Langevin_step'],
                 Langevin_eta=config['Langevin_eta'],
                 is_anneal_Langevin=True,
                 Langevin_adjust_step=config['Langevin_adjust_step'],
                 deep_hidden_sizes=config['deep_hidden_sizes'],
                 is_cuda=config['cuda'],)

    if config['cuda']:
        model.cuda()

    param_wd, param_no_wd = [], []
    for xx, yy in model.named_parameters():
        if 'W' in xx:
            param_wd += [yy]
        else:
            param_no_wd += [yy]

    optimizer = optim.SGD([{
        'params': param_no_wd,
        'weight_decay': 0
    }, {
        'params': param_wd
    }],
        lr=config['lr'],
        momentum=0.0,
        weight_decay=config['wd'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['epochs'])

    if config['resume'] > 0:
        load(model, config['exp_folder'], config['resume'])

    for epoch in range(config['resume']):
        scheduler.step()

    is_show_training_data = False
    model.go_deep = False
    for epoch in range(config['resume'] + 1, config['epochs'] + 1):
        if epoch <= config['Langevin_adjust_warmup_epoch']:
            model.set_Langevin_adjust_step(config['CD_step'])
        else:
            model.set_Langevin_adjust_step(config['Langevin_adjust_step'])

        recon_loss = train(model,
                           train_loader,
                           optimizer,
                           config)

        var = model.get_var().detach().cpu().numpy()

        # show samples periodically
        if epoch % config['log_interval'] == 0:
            if 'GMM' in config['dataset']:
                logger.info(
                    f'PID={pid} || {epoch} epoch || mean = {model.mu.detach().cpu().numpy()} || var={model.get_var().detach().cpu().numpy()} || Reconstruction Loss = {recon_loss}'
                )
            else:
                logger.info(
                    f'PID={pid} || {epoch} epoch || var={model.get_var().mean().item()} || Reconstruction Loss = {recon_loss}'
                )
                wandb.log({"reconstruction_loss": recon_loss, "var": var}, commit=False)
                if test_loader is not None:
                    visualize_sampling(model, epoch, config, is_show_gif=False, test_loader=test_loader, shortcut_mse_calculation=True)

        if epoch % config['vis_interval'] == 0 or epoch == config['epochs']:
            visualize_sampling(model,
                                epoch,
                                config,
                                is_show_gif=True,
                                test_loader=test_loader)

            # visualize one mini-batch of training data
            if not is_show_training_data and 'GMM' not in config['dataset']:
                data, _ = next(iter(train_loader))
                mean = config['img_mean'].view(1, -1, 1, 1).to(data.device)
                std = config['img_std'].view(1, -1, 1, 1).to(data.device)
                vis_data = (data * std + mean).clamp(min=0, max=1)
                utils.save_image(
                    utils.make_grid(vis_data,
                                    nrow=config['sampling_nrow'],
                                    normalize=False,
                                    padding=1,
                                    pad_value=1.0).cpu(),
                    f"{config['exp_folder']}/training_imgs.png")
                is_show_training_data = True

            # visualize filters & hidden states
            if config['is_vis_verbose']:
                filters = model.W.T.view(model.W.shape[1], config['channel'],
                                            config['height'], config['width'])
                utils.save_image(
                    filters,
                    f"{config['exp_folder']}/filters_epoch_{epoch:05d}.png",
                    nrow=8,
                    normalize=True,
                    padding=1,
                    pad_value=1.0)

                # visualize hidden states
                data, _ = next(iter(train_loader))
                h_pos = model.prob_h_given_v(
                    data.view(data.shape[0], -1).cuda(), model.get_var())
                utils.save_image(h_pos.view(1, 1, -1, config['hidden_size']),
                                    f"{config['exp_folder']}/hidden_epoch_{epoch:05d}.png",
                                    normalize=True)

        # save models periodically
        if epoch == config['epochs']:
            save(model, config['exp_folder'], epoch)

        scheduler.step()

    if config['deep_hidden_sizes'] is not None:
        # now we can train rest of the network freezing the first layer GRBM
        
        data_loader = DataLoaderWrapper(train_loader, transform=lambda x: torch.bernoulli(model.prob_h_given_v(x, model.get_var())))
        for i in range(len(model.deep_rbms)):
            # train one RBM layer at a time 
            rbm = model.deep_rbms[i]
            logger.info(f'Training Bernoulli RBM {rbm.visible_size} -> {rbm.hidden_size}')
            epoch_losses = train_rbm(rbm, data_loader, 'cuda' if config['cuda'] else 'cpu')
            loss_data = {'epoch_{}'.format(i+1): loss for i, loss in enumerate(epoch_losses)}
            wandb.log(loss_data)
            logger.info(f'Loss over epochs: {epoch_losses}')
            if i < len(model.deep_rbms) - 1:
                data_loader = DataLoaderWrapper(data_loader, rbm.visible_to_hidden)

    if len(model.deep_rbms) > 0:
        model.go_deep = True

    if model.go_deep:
        # now we can fine tune the first layer since all the other layers are trained
        start_epoch = config['epochs'] + 2
        end_epoch = start_epoch + (config['epochs'] // 2)

        for epoch in range(start_epoch, end_epoch):
            model.set_Langevin_adjust_step(config['Langevin_adjust_step'])

            recon_loss = train(model,
                            train_loader,
                            optimizer,
                            config)

            var = model.get_var().detach().cpu().numpy()

            # show samples periodically
            if epoch % config['log_interval'] == 0:
                if 'GMM' in config['dataset']:
                    logger.info(
                        f'PID={pid} || {epoch} epoch || mean = {model.mu.detach().cpu().numpy()} || var={model.get_var().detach().cpu().numpy()} || Reconstruction Loss = {recon_loss}'
                    )
                else:
                    logger.info(
                        f'PID={pid} || {epoch} epoch || var={model.get_var().mean().item()} || Reconstruction Loss = {recon_loss}'
                    )
                    wandb.log({"reconstruction_loss": recon_loss, "var": var}, commit=False)
                    if test_loader is not None:
                        visualize_sampling(model, epoch, config, is_show_gif=False, test_loader=test_loader, shortcut_mse_calculation=True)

            if epoch % config['vis_interval'] == 0:
                visualize_sampling(model,
                                    epoch,
                                    config,
                                    is_show_gif=True,
                                    test_loader=test_loader)

                # visualize one mini-batch of training data
                if not is_show_training_data and 'GMM' not in config['dataset']:
                    data, _ = next(iter(train_loader))
                    mean = config['img_mean'].view(1, -1, 1, 1).to(data.device)
                    std = config['img_std'].view(1, -1, 1, 1).to(data.device)
                    vis_data = (data * std + mean).clamp(min=0, max=1)
                    utils.save_image(
                        utils.make_grid(vis_data,
                                        nrow=config['sampling_nrow'],
                                        normalize=False,
                                        padding=1,
                                        pad_value=1.0).cpu(),
                        f"{config['exp_folder']}/training_imgs.png")
                    is_show_training_data = True

                # visualize filters & hidden states
                if config['is_vis_verbose']:
                    filters = model.W.T.view(model.W.shape[1], config['channel'],
                                                config['height'], config['width'])
                    utils.save_image(
                        filters,
                        f"{config['exp_folder']}/filters_epoch_{epoch:05d}.png",
                        nrow=8,
                        normalize=True,
                        padding=1,
                        pad_value=1.0)

                    # visualize hidden states
                    data, _ = next(iter(train_loader))
                    h_pos = model.prob_h_given_v(
                        data.view(data.shape[0], -1).cuda(), model.get_var())
                    utils.save_image(h_pos.view(1, 1, -1, config['hidden_size']),
                                        f"{config['exp_folder']}/hidden_epoch_{epoch:05d}.png",
                                        normalize=True)

            # save models periodically
            if epoch == end_epoch - 1:
                save(model, config['exp_folder'], epoch)

        visualize_sampling(model,
                            99999,
                            config,
                            is_show_gif=True,
                            test_loader=test_loader,
                            after_finetune=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        help='Dataset name {gmm_iso, gmm_aniso, mnist, fashionmnist, celeba, celeba2K}')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--disable_deep', action='store_true', default=False)
    args = parser.parse_args()

    train_model(args)
    wandb.finish()
