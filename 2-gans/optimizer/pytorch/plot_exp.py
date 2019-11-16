import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

learning_rates = [0.5, 1e-2, 1e-5]
optimizers = ['sgd', 'momentumsgd', 'rmsprop', 'adam', 'adagrad']

outdir = 'output'

for lr in learning_rates:
    loss = pd.DataFrame(columns=['Optimizer', 'Training loss', 'Epochs'])
    acc = pd.DataFrame(columns=['Optimizer', 'Training accuracy', 'Epochs'])

    for opt in optimizers:
        for i in range(3):
            for j, val in enumerate(pickle.load(open(outdir+'/'+opt+str(lr)+str(i)+".pkl", "rb"))['train_loss']):
                loss = loss.append({'Optimizer': opt, 'Epochs': j, 'Training loss': val}, ignore_index=True)
            
            for j, val in enumerate(pickle.load(open(outdir+'/'+opt+str(lr)+str(i)+".pkl", "rb"))['train_accuracy']):
                acc = acc.append({'Optimizer': opt, 'Epochs': j, 'Training accuracy': val}, ignore_index=True)

    ax1 = sns.lineplot(x='Epochs', y='Training loss', hue='Optimizer', data=loss)
    ax1.set_title('Learning rate =' + str(lr))
    plt.savefig('loss_'+str(lr)+'.pdf')
    plt.show()

    ax2 = sns.lineplot(x='Epochs', y='Training accuracy', hue='Optimizer', data=acc)
    ax2.set_title('Learning rate =' + str(lr))
    plt.savefig('acc_'+str(lr)+'.pdf')
    plt.show()

