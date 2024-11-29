import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

class OpticalFlow(torch.nn.Module):
    
    def __init__(self, mask_mode="both") -> None:
        super().__init__()
        self.of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval()
        self.grid = None
        self.grid_norm = None
        assert mask_mode in ["color", "flow", "both"]
        self.mask_mode = mask_mode
    
    def get_flows(self, a_images, b_images, get_reverse=False, get_mask=False, flow_mask_epsilon=1.0, color_mask_epsilon=0.1):
        
        if get_reverse or get_mask:
            flows = self.of_model(torch.cat([a_images, b_images]), torch.cat([b_images, a_images]))[-1]
            ab_flows, ba_flows = flows[:a_images.size(0)], flows[a_images.size(0):]
        else:
            ab_flows = self.of_model(a_images, b_images)[-1]
            return ab_flows
        
        if get_mask:
            
            ba_flows_registered, b_registered = self.grid_sample([ba_flows, b_images], ab_flows)
            loop_displacement = (ab_flows + ba_flows_registered).pow(2).sum(dim=1, keepdim=True).sqrt() < flow_mask_epsilon
            color_mask = (b_registered - a_images).pow(2).sum(dim=1, keepdim=True).sqrt() < color_mask_epsilon
            
            if self.mask_mode == "both":
                mask = loop_displacement * color_mask
            elif self.mask_mode == "color":
                mask = color_mask
            elif self.mask_mode == "flow":
                mask = loop_displacement
            
            if get_reverse:
                return ab_flows, ba_flows, mask
            else:
                return ab_flows, mask
            
        return ab_flows, ba_flows
        
    def grid_sample(self, images, flows):
        sizes = None
        if isinstance(images, list) or isinstance(images, tuple):
            sizes = [i.size(1) for i in images]
            images = torch.cat(images, dim=1)
        grid, grid_norm = self.__get_grid(flows)
        grid = grid + flows
        grid = torch.mul(grid, grid_norm[None, :, None, None]) - 1
        sampled = torch.nn.functional.grid_sample(images, grid.permute(0, 2, 3, 1), "bilinear", align_corners=True)
        if sizes != None:
            starts = [0] + torch.cumsum(torch.tensor(sizes), 0).tolist()[:-1]
            return [sampled[:, s:s+c] for s, c in zip(starts, sizes)]
        else:
            return sampled
    
    def __get_grid(self, flows):
        if self.grid == None or self.grid.shape != flows.shape:
            b, _, h, w = flows.shape
            xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
            yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
            xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
            yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
            self.grid = torch.cat((xx, yy), 1).float().to(flows.device)
            self.grid_norm = 2.0 / (torch.clamp(torch.tensor([w, h]) - 1, 1)).to(flows.device)
        return self.grid, self.grid_norm
