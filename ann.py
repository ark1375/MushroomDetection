import enum
import torch
from torch import nn

device = 'cpu'

def readDS(location):
    
    loaded_data = list()

    with open(location , 'r') as dataset:
        while(data := dataset.readline()):
            d_split = data.split(',')
            d_int = list ( map(int , d_split) )
            loaded_data.append(
                ([d_int[0]] , d_int[1:])
            )

    return loaded_data

# for d in readDS('converted_ds.data')[:10]:
#     print(d , end='\n')

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(21 , 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.linear_stack(x)
        return out


def train(ds , model , loss_fn , optimizer):
    model.train()

    avg_loss = 0


    b_loss = 0
    b_s = 20
    for indx , (Y , X) in enumerate(ds):

        X = torch.FloatTensor(X)

        if Y == [0]:
            Y = torch.FloatTensor([1,0])
        else:
            Y = torch.FloatTensor([0,1])

        pred = model(X)
        loss = loss_fn(pred , Y)

        # if indx % b_s == 1:
        #     b_loss /= b_s
        #     optimizer.zero_grad()
        #     b_loss.backward()
        #     optimizer.step()

        # if indx % 20 == 0:
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if indx % 100 == 0:
        
        # else:
        #     b_loss += loss


        
        
        if indx % 100 == 0:
            print(f"loss: {loss:>7f}" , end='\r')
        
        avg_loss += loss

    avg_loss /= len(ds)

    print(f"Average Loss: {avg_loss:>7f}" , end='\n')

def test(ds , model , loss_fn):
    model.eval()
    test_loss , correct = 0 , 0
    with torch.no_grad():
        for (Y , X) in ds:

            X = torch.FloatTensor(X)
            y = Y[0]
            if (Y == [0]):
                Y = torch.FloatTensor([1,0])
            else:
                Y = torch.FloatTensor([0,1])

            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            pred = pred.argmax(-1).item()


            correct += int(pred == y)

    test_loss /= len(ds)
    correct /= len(ds)

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():

    dataset = readDS('converted_ds.data')

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD( model.parameters() , lr=1e-3)

    for t in range(epochs:=100):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataset[0:7000 ], model, loss_fn, optimizer)
        test(dataset[7000:], model, loss_fn)
        print('===============================\n') 


if __name__ == '__main__':
    main()