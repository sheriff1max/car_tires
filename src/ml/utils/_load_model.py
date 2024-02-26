import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path_model: str):
    model = torch.load(path_model, map_location=device)
    model.eval()
    return model
