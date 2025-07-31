import torch

def vectorize_model(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

def update_model(model, vector):
    torch.nn.utils.vector_to_parameters(vector, model.parameters())
