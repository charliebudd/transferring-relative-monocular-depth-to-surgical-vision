import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from src.metamed import MetaMED, ssimae
from src.trmdsv import load_model

def main(args):

    model, spatial_transform, color_transform = load_model(args.model, args.weights, "cuda")
    model.eval()

    datasets = {
        "scared": MetaMED(args.data_directory, "Testing-SCARED"),
        "serv-ct": MetaMED(args.data_directory, "Testing-SERVCT"),
    }

    all_errors = []

    for d in datasets:

        dataset = datasets[d]
        dataloader = DataLoader(dataset, 4, True, num_workers=4, pin_memory=True)

        errors = []

        for sample in dataloader:

            images, target_depths = sample[0].cuda(), sample[1].cuda()

            with torch.no_grad():
                predicted_depths = model(color_transform(spatial_transform(images))).unsqueeze(1)
            
            predicted_depths = resize(predicted_depths, target_depths.shape[-2:])
            maes = ssimae(predicted_depths, target_depths, ~torch.isnan(target_depths))
            
            errors += maes.tolist()

        all_errors += errors

        mean = sum(errors) / len(errors)
        print(f"{d}: {mean:0.3f}")

    mean = sum(all_errors) / len(all_errors)
    print(f"overall: {mean:0.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=str)
    parser.add_argument("--model", type=str, choices=["midas", "depthanything"])
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()

    main(args)