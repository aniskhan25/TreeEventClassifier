def train(loader, model, optimizer, criterion):
    model.train()

    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)

        target = batch.y.float().unsqueeze(1)

        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
