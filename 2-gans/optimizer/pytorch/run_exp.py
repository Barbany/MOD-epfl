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

