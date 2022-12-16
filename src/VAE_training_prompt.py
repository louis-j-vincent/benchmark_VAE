# Install the library
#pip install git+https://github.com/louis-j-vincent/benchmark_VAE.git@missing_values

import torch
import torchvision.datasets as datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

#download datasets
mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)

train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

config = BaseTrainerConfig(
    output_dir='my_model',
    learning_rate=1e-4,
    batch_size=100,
    num_epochs=10, # Change this to train the model a bit more
)


model_config = AEConfig(
    input_dim=(1, 28, 28),
    latent_dim=16
)

model = AE(
    model_config=model_config,
    encoder=Encoder_AE_MNIST(model_config), 
    decoder=Decoder_AE_MNIST(model_config) 
)
model.n = 10

pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset
)

torch.save(model.state_dict(), 'model_state_dict')