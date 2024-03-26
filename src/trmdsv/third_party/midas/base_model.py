import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))
        
        keys_to_delete = [key for key in parameters if "relative_position_index" in key]
        for key in keys_to_delete:
            del parameters[key]

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
