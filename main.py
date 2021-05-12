from optimizers import lenet5, lenet5_sgdm, lenet5_adam, lenet5_adagrad
from optimizers.lenet5 import run_sgd
from optimizers.lenet5_sgdm import run_sgdm
from optimizers.lenet5_adagrad import run_adagrad
from optimizers.lenet5_adam import run_adam
from optimizers.lenet5_rmsprop import run_rms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
def get_table_data(model):
    if model == 'SGD':
        lr, acc, loss, t = run_sgd()

    elif model == 'SGDM':
        lr, acc, loss, t = run_sgdm()
        
    elif model == 'Adagrad':
        lr, acc, loss, t = run_adagrad()
        
    elif model == 'Adam':
        lr, acc, loss, t = run_adam()

    elif model == 'RMS':
        lr, acc, loss, t = run_rms()

    table_data = []

    for i in range(len(lr)):
        table_data.append([lr[i], acc[i], loss[i], t[i]])
        
    return table_data, model

def display_table(table_data, model):
    fig, ax = plt.subplots()
    table = ax.table(cellText=table_data, loc='center', cellLoc='left', colLabels=['Learning Rate', 'Test Accuracy (%)', 'Test Loss', 'Training Time (s)'])
    table.scale(1, 4)
    table.auto_set_font_size(False)
    # table.set_fontsize(14)
    ax.axis('off')
    plt.savefig('results/' + model + '_results.jpg')
    # plt.show()

def display_all():
    print("SGD\n\n")
    lr, sgd_acc, sgd_loss, sgd_t = run_sgd()

    table_data = []

    for i in range(len(lr)):
        table_data.append([lr[i], sgd_acc[i], sgd_loss[i], sgd_t[i]])

    display_table(table_data, 'SGD')

    print("SGDM\n\n")
    lr, sgdm_acc, sgdm_loss, sgdm_t = run_sgdm()
    table_data = []

    for i in range(len(lr)):
        table_data.append([lr[i], sgdm_acc[i], sgdm_loss[i], sgdm_t[i]])

    display_table(table_data, 'SGDM')

    print("AdaGrad\n\n")
    lr, ada_acc, ada_loss, ada_t = run_adagrad()
    table_data = []

    for i in range(len(lr)):
        table_data.append([lr[i], ada_acc[i], ada_loss[i], ada_t[i]])

    display_table(table_data, 'AdaGrad')

    print('Adam\n\n')
    lr, adam_acc, adam_loss, adam_t = run_adam()
    table_data = []

    for i in range(len(lr)):
        table_data.append([lr[i], adam_acc[i], adam_loss[i], adam_t[i]])

    display_table(table_data, 'Adam')

    print('RMS\n\n')
    lr, rms_acc, rms_loss, rms_t = run_rms()    
    table_data = []

    for i in range(len(lr)):
        table_data.append([lr[i], rms_acc[i], rms_loss[i], rms_t[i]])

    display_table(table_data, 'RMS')

    fig, ax = plt.subplots()
    ax.plot(lr, sgd_acc, label='SGD', marker='o', color='blue')
    ax.plot(lr, sgdm_acc, label="SGDM", marker='o')
    ax.plot(lr, ada_acc, label='AdaGrad', marker='o')
    ax.plot(lr, adam_acc, label='Adam', marker='o')
    ax.plot(lr, rms_acc, label='RMS', marker='o')
    ax.set_title('Test Accuracy')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Test Accuracy")
    ax.legend(labels=['SGD', 'SGDM', 'AdaGrad', 'Adam', 'RMS'])
    plt.savefig('results/accuracy.jpg')

    fig, ax = plt.subplots()
    ax.plot(lr, sgd_loss, label='SGD', marker='o')
    ax.plot(lr, sgdm_loss, label="SGDM", marker='o')
    ax.plot(lr, ada_loss, label='AdaGrad', marker='o')
    ax.plot(lr, adam_loss, label='Adam', marker='o')
    ax.plot(lr, rms_loss, label='RMS', marker='o')
    ax.set_title('Test Loss')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel("Test Loss")
    ax.legend(labels=['SGD', 'SGDM', 'AdaGrad', 'Adam', 'RMS'])
    plt.savefig('results/loss.jpg')

    fig, ax = plt.subplots()
    ax.plot(lr, sgd_t, label='SGD', marker='o')
    ax.plot(lr, sgdm_t, label="SGDM", marker='o')
    ax.plot(lr, ada_t, label='AdaGrad', marker='o')
    ax.plot(lr, adam_t, label='Adam', marker='o')
    ax.plot(lr, rms_t, label='RMS', marker='o')
    ax.set_title('Training and Testing Time (s)')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Training and Testing Time (s)')
    ax.legend(labels=['SGD', 'SGDM', 'AdaGrad', 'Adam', 'RMS'])
    plt.savefig('results/train_time.jpg')


if __name__ == '__main__':
    print("1. SGD\n2. SGD w/ Momentum (SGDM)\n3. Adagrad\n4. Adam\n5. RMSProp (RMS)\n6. All")
    model = input("Enter the model to train: ")
    if model != 'All':
        table_data, model = get_table_data(model)
        display_table(table_data, model)
    else:
        display_all()
       