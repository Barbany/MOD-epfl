import matplotlib.pyplot as plt
import pickle
import os

outdir='output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

learning_rates = [0.5, 1e-2, 1e-5]

# TODO: Loop 3 times and average results
for lr in learning_rates:
    os.system('python main.py --optimizer sgd --learning_rate '+str(lr)+' --output='+outdir+'/sgd'+str(lr)+'.pkl')
    os.system('python main.py --optimizer momentumsgd --learning_rate '+str(lr)+' --output='+outdir+'/momentumsgd'+str(lr)+'.pkl')
    os.system('python main.py --optimizer rmsprop --learning_rate '+str(lr)+' --output='+outdir+'/rmsprop'+str(lr)+'.pkl')
    os.system('python main.py --optimizer adam --learning_rate '+str(lr)+' --output='+outdir+'/adam'+str(lr)+'.pkl')
    os.system('python main.py --optimizer adagrad --learning_rate '+str(lr)+' --output='+outdir+'/adagrad'+str(lr)+'.pkl')
    optimizers = ['sgd', 'momentumsgd', 'rmsprop', 'adam', 'adagrad']

    # Plots the training losses.
    for optimizer in optimizers:
       data = pickle.load(open(outdir+'/'+optimizer+str(lr)+".pkl", "rb"))
       plt.plot(data['train_loss'], label=optimizer)
    plt.ylabel('Trainig loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('loss_'+str(lr)+'.pdf')
    plt.show()

    # Plots the training accuracies.
    for optimizer in optimizers:
        data = pickle.load(open(outdir+'/'+optimizer+str(lr)+".pkl", "rb"))
        plt.plot(data['train_accuracy'], label=optimizer)
    plt.ylabel('Trainig accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('accuracy_'+str(lr)+'.pdf')
    plt.show()

