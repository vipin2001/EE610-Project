import numpy as np
import time
import torch
import matplotlib.pyplot as plt

def train(model, opt, criterion, train_loader, test_loader, model_name, color_space, num_epochs=10, stdout_frequency=10):
    print("============================== TRAINING STARTED ==============================")

    f = open(f"{model_name}-{color_space}.log", 'w')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    train_loss = []
    test_loss = []
    test_acc = []
    total_steps = []
    total_loss = 0
    iterations = 0
    start = time.time()
    for e in range(num_epochs):
        for inputs, labels in train_loader:
            print(iterations)
            iterations += 1
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

            if iterations % stdout_frequency == 0:
                current_loss = 0
                acc = 0
                model.eval() 
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        current_loss += criterion(outputs, labels).item()

                        probs = torch.exp(outputs)
                        _, c = probs.topk(1, dim=1)
                        correct = c == labels.view(*c.shape)
                        acc += torch.mean(correct.type(torch.FloatTensor)).item()

                train_loss.append(total_loss/stdout_frequency)
                test_loss.append(current_loss/len(test_loader))
                test_acc.append(acc/len(test_loader))
                total_steps.append(iterations)
                current = time.time()
                print("Time Elapsed:", current-start)
                start = current
                f.write(f"epoch: {e+1},\titeration: {iterations},\ttrain loss: {(total_loss/stdout_frequency):.3f},\ttest loss: {current_loss/len(test_loader):.3f},\ttest accuracy: {acc/len(test_loader):.3f}\n")
                print(f"epoch: {e+1},\titeration: {iterations},\ttrain loss: {(total_loss/stdout_frequency):.3f},\ttest loss: {current_loss/len(test_loader):.3f},\ttest accuracy: {acc/len(test_loader):.3f}")
                total_loss = 0  
                model.train()


    checkpoint = {'parameters': model.parameters, 'state_dict': model.state_dict}
    torch.save(checkpoint, f'{model_name}-{color_space}.pth')

    plt.plot(total_steps, train_loss, label='Train Loss')
    plt.plot(total_steps, test_loss, label='Test Loss')
    plt.plot(total_steps, test_acc, label='Test Accuracy')
    plt.legend()
    plt.title(f'{model_name}-{color_space}')
    plt.xlabel('Iterations')
    plt.grid()
    plt.savefig(f'{model_name}-{color_space}.png')
    f.close()


                        


            
