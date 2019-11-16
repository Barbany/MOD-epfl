import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

outdir='output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

learning_rates = [0.5, 1e-2, 1e-5]

for lr in learning_rates:
    for i in range(3):
        os.system('python main.py --optimizer sgd --learning_rate '+str(lr)+' --output='+outdir+'/sgd'+str(lr)+str(i)+'.pkl')
        os.system('python main.py --optimizer momentumsgd --learning_rate '+str(lr)+' --output='+outdir+'/momentumsgd'+str(lr)+str(i)+'.pkl')
        os.system('python main.py --optimizer rmsprop --learning_rate '+str(lr)+' --output='+outdir+'/rmsprop'+str(lr)+str(i)+'.pkl')
        os.system('python main.py --optimizer adam --learning_rate '+str(lr)+' --output='+outdir+'/adam'+str(lr)+str(i)+'.pkl')
        os.system('python main.py --optimizer adagrad --learning_rate '+str(lr)+' --output='+outdir+'/adagrad'+str(lr)+str(i)+'.pkl')
    
    optimizers = ['sgd', 'momentumsgd', 'rmsprop', 'adam', 'adagrad']

    # Plots the training losses.
    for optimizer in optimizers:
        # Average train loss in the 3 runs
        train_loss = 0
        for i in range(3):
            train_loss += np.asarray(pickle.load(open(outdir+'/'+optimizer+str(lr)+str(i)+".pkl", "rb"))['train_accuracy']) / 3
        plt.plot(train_loss, label=optimizer)

    plt.ylabel('Trainig loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('loss_'+str(lr)+'.pdf')
    plt.show()

    # Plots the training accuracies.
    for optimizer in optimizers:
        # Average train loss in the 3 runs
        train_acc = 0
        for i in range(3):
            train_acc += np.asarray(pickle.load(open(outdir+'/'+optimizer+str(lr)+str(i)+".pkl", "rb"))['train_accuracy']) / 3
        plt.plot(train_acc, label=optimizer)
    
    plt.ylabel('Trainig accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('accuracy_'+str(lr)+'.pdf')
    plt.show()

