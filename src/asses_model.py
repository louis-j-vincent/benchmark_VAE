import sys

model_path = sys.argv[1]
N = sys.argv[2]

model = 

model.to('cpu')
model.n = 10
x = train_dataset[:1000].detach()
z = model.encoder(x).embedding
x_rec = model.decoder(z).reconstruction