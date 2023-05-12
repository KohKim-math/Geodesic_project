import torch
from tqdm import tqdm
from utils import EarlyStopping

def train(model, train_loader, valid_loader, num_epochs, patience, model_directory):
    model_name = 'model'
    early_stopping = EarlyStopping(patience = patience, verbose = False, path=f'{model_directory}/checkpoint.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-3, weight_decay=10e-6)

    for epoch in range(num_epochs):
        # train step
        model.train()
        train_loss = 0.
        # for batch in tqdm(train_loader):
        for batch in train_loader:
            data = batch[0].float()
            optimizer.zero_grad()
            loss, _ = model(data)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(train_loader.dataset)
        train_loss *= 1000.
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.item():.4f}')

        # validation step
        valid_loss = 0.
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                data = batch[0].float()
                loss, _ = model(data)
                valid_loss += loss
        valid_loss /= len(valid_loader.dataset)
        valid_loss *= 1000.

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            epoch = epoch - patience
            print(f"Early stopping at {epoch}")
            break

    model.load_state_dict(torch.load(f'{model_directory}/checkpoint.pt'))

    torch.save({
                'model_state_dict': model.state_dict(),
            },  f'{model_directory}/{model_name}.pth.tar')


