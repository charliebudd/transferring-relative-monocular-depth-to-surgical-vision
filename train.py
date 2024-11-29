import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from torchvision.transforms import ColorJitter, RandomGrayscale, RandomHorizontalFlip, RandomAffine
from torchvision.transforms.functional import resize
from argparse import ArgumentParser

from src.metamed import MetaMED, JointTransform, ssimae
from src.opticalflow import OpticalFlow
from src.trmdsv import load_model

def train(args):

    device = torch.device("cuda")

    # So that an epoch is roughly the same amount of data
    args.epoch_batches = int(args.epoch_batches / len(args.train_mode))
    do_sup = "sup" in args.train_mode
    do_aug = "aug" in args.train_mode
    do_temp = "temp" in args.train_mode

    # Output directory...
    out_dir = f"outputs/{args.name}"
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) != 0:
        resuming = True
    else:
        resuming = False
        os.makedirs(out_dir, exist_ok=True)

    # Model preparation...
    optical_flow = OpticalFlow().to(device)
    depth_model, spatial_transform, color_transform = load_model(args.model, None, device)
    depth_model.train().requires_grad_(True).to(device)
    
    # Transforms
    pre_model = Compose([spatial_transform, color_transform])
    pre_flow = Compose([Resize((384, 384), antialias=True), Normalize(0.5, 0.5)])
    colour_augmentation = Compose([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        RandomGrayscale(p=0.1),
    ])
    spatial_augmentation = Compose([
        RandomHorizontalFlip(),
        RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15, interpolation=InterpolationMode.BILINEAR),
    ])
    jount_augmentation = JointTransform(spatial_augmentation, colour_augmentation)
    
    if do_temp or do_aug:
        depth_model_slow, _, _ = load_model(args.model, None, device)
        depth_model_slow.eval().requires_grad_(False).to(device)
    
    # Loss and optimiser...
    optimiser = torch.optim.SGD(depth_model.parameters(), args.learning_rate, args.momentum)

    # Datasets...
    if do_sup:
        sup_dataset = MetaMED(args.data_directory, "Training", None, None)
        sup_dataloader = DataLoader(sup_dataset, args.batch_size, True, num_workers=4, pin_memory=True)
   
    if do_aug:
        aug_dataset = MetaMED(args.data_directory, "Unlabeled", spatial_transform, None)
        aug_dataloader = DataLoader(aug_dataset, args.batch_size, True, num_workers=4, pin_memory=True)
        
    if do_temp:
        temp_dataset = MetaMED(args.data_directory, "UnlabeledPairs", spatial_transform, None)
        temp_dataloader = DataLoader(temp_dataset, args.batch_size, True, num_workers=4, pin_memory=True)

    val_dataset = MetaMED(args.data_directory, "Validation", None, None)
    val_dataloader = DataLoader(val_dataset, args.batch_size, False, num_workers=4, pin_memory=True)

    # Resuming...
    if resuming:
        depth_model.load_state_dict(torch.load(f"{out_dir}/latest_weights.pt", device))
        if do_temp or do_aug:
            depth_model_slow.load_state_dict(torch.load(f"{out_dir}/latest_slow_weights.pt", device))
        if do_sup:
            sup_losses = torch.load(f"{out_dir}/sup_losses.pt")
        if do_aug:
            aug_losses = torch.load(f"{out_dir}/aug_losses.pt")
        if do_temp:
            temp_losses = torch.load(f"{out_dir}/temp_losses.pt")
        validation_losses = torch.load(f"{out_dir}/validation_losses.pt")
    else:
        sup_losses = []
        aug_losses = []
        temp_losses = []
        validation_losses = []

    # Training Loop...
    for epoch in range(len(validation_losses), args.max_epoch):

        # Training...
        depth_model.train()
        epoch_sup_losses = []
        epoch_aug_losses = []
        epoch_temp_losses = []
        data_iter = zip(
            range(args.epoch_batches),
            sup_dataloader if do_sup else args.epoch_batches * [None],
            aug_dataloader if do_aug else args.epoch_batches * [None],
            temp_dataloader if do_temp else args.epoch_batches * [None],
        )
        for batch_index, sup_data, aug_data, temp_data in data_iter:
            
            #####################################################
            if do_sup:
                images, target_depths = sup_data[0].cuda(), sup_data[1].cuda()
                
                images = colour_augmentation(images)
                
                predicted_depths = depth_model(color_transform(spatial_transform(images))).unsqueeze(1)
                predicted_depths = resize(predicted_depths, target_depths.shape[-2:])         
                       
                sup_loss = ssimae(predicted_depths, target_depths, ~torch.isnan(target_depths)).mean()
            
                # Update model weights...
                optimiser.zero_grad()
                sup_loss.backward()
                if args.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(depth_model.parameters(), args.gradient_clipping)
                optimiser.step()

            #####################################################
            if do_aug:
                image = aug_data.cuda()
                
                with torch.no_grad():
                    target_depths = depth_model_slow(color_transform(image)).unsqueeze(1)
                    target_depths = (target_depths - target_depths.flatten(1).mean(1)[:, None, None, None]) / target_depths.flatten(1).std(1)[:, None, None, None]
                    
                aug_image, target_depths = jount_augmentation(image, depth=target_depths)
                mask = aug_image.sum(dim=1, keepdim=True) != 0
                
                predicted_depths = depth_model(color_transform(aug_image)).unsqueeze(1)
                aug_loss = ssimae(predicted_depths, target_depths, mask, normalise=False).mean()
                
                # Update model weights...
                optimiser.zero_grad()
                aug_loss.backward()
                if args.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(depth_model.parameters(), args.gradient_clipping)
                optimiser.step()
                
            #####################################################
            if do_temp:
                a_images, b_images = temp_data[0].cuda(), temp_data[1].cuda()
                
                with torch.no_grad():
                    b_depths = depth_model_slow(color_transform(b_images)).unsqueeze(1)
                    b_depths = (b_depths - b_depths.flatten(1).mean(1)[:, None, None, None]) / b_depths.flatten(1).std(1)[:, None, None, None]
                    ab_flows = optical_flow.get_flows(pre_flow(a_images), pre_flow(b_images))
                    ab_flows = resize(ab_flows, b_depths.shape[-2:])
                    b_depths_registered = optical_flow.grid_sample(b_depths, ab_flows)
                
                a_images, b_depths_registered = jount_augmentation(a_images, depth=b_depths_registered)
                mask = a_images.sum(dim=1, keepdim=True) != 0
                    
                a_depths = depth_model(color_transform(a_images)).unsqueeze(1)
                temp_loss = ssimae(a_depths, b_depths_registered, mask, normalise=False)
                temp_loss = temp_loss[~torch.isnan(temp_loss)].mean()
                
                # Update model weights...
                optimiser.zero_grad()
                temp_loss.backward()
                if args.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(depth_model.parameters(), args.gradient_clipping)
                optimiser.step()
            
            #####################################################
            # Update slow model weights via exponential moving average...
            if do_temp or do_aug:
                if batch_index % args.ema_step == 0:
                    with torch.no_grad():
                        slow_params, fast_params = depth_model_slow.parameters(), depth_model.parameters()
                        for slow_param, fast_param in zip(slow_params, fast_params):
                            slow_param.data.copy_(args.ema_gamma * slow_param.data + (1 - args.ema_gamma) * fast_param.data)
           
            # Track running losses...
            if do_sup:
                epoch_sup_losses.append(sup_loss.item())
            if do_aug:
                epoch_aug_losses.append(aug_loss.item())
            if do_temp:
                epoch_temp_losses.append(temp_loss.item())

        if do_sup:
            sup_losses.append(sum(epoch_sup_losses) / len(epoch_sup_losses))
        if do_aug:
            aug_losses.append(sum(epoch_aug_losses) / len(epoch_aug_losses))
        if do_temp:
            temp_losses.append(sum(epoch_temp_losses) / len(epoch_temp_losses))
                
        # Validation...
        depth_model.eval()
        running_losses = None
        with torch.no_grad():
            for images, depths in val_dataloader:
                images, depths = images.cuda(), depths.cuda()
                
                predicted_depths = depth_model(pre_model(images)).unsqueeze(1)
                
                predicted_depths = resize(predicted_depths, depths.shape[-2:])
                ssimaes = ssimae(predicted_depths, depths, ~torch.isnan(depths))
                
                running_losses = ssimaes if running_losses == None else torch.cat([running_losses, ssimaes]) 
                
        validation_loss = sum(running_losses) / len(running_losses)
        validation_losses.append(validation_loss)
        
        # Saving and Logging...
        print(f"Epoch {epoch:04d}: validation_loss: {validation_loss:0.3f}")
        state_dict = depth_model.state_dict()
        torch.save(state_dict, f"{out_dir}/latest_weights.pt")
        if do_temp or do_aug:
            state_dict = depth_model_slow.state_dict()
            torch.save(state_dict, f"{out_dir}/latest_slow_weights.pt")
        if validation_losses[-1] == min(validation_losses):
            shutil.copyfile(f"{out_dir}/latest_weights.pt", f"{out_dir}/best_weights_validation.pt")
        if do_aug:
            torch.save(aug_losses, f"{out_dir}/aug_losses.pt")
        if do_temp:
            torch.save(temp_losses, f"{out_dir}/temp_losses.pt")
        if do_sup:
            torch.save(sup_losses, f"{out_dir}/sup_losses.pt")
        torch.save(validation_losses, f"{out_dir}/validation_losses.pt")

        if epoch - torch.argmin(torch.tensor(validation_losses)) > args.early_stop:
            print("Training finished!")
            break


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model", type=str, choices=["midas", "depthanything"])
    parser.add_argument("--train-mode", type=str, nargs="*", choices=["sup", "aug", "temp"])
    parser.add_argument("--data-directory", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--gradient-clipping", type=float, default=10)
    parser.add_argument("--ema-step", type=int, default=5)
    parser.add_argument("--ema-gamma", type=float, default=0.95)
    parser.add_argument("--epoch-batches", type=int, default=60)
    parser.add_argument("--max-epoch", type=int, default=750)
    parser.add_argument("--early-stop", type=int, default=100)
    args = parser.parse_args()
    
    train(args)
