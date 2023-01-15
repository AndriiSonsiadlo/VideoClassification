"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        losses = []
        accuracies = []
        top_5_accuracies = []
        cnn_benchmark = []  # this is ridiculous
        for epoch,acc,loss,top_k_categorical_accuracy,val_acc,val_loss,val_top_k_categorical_accuracy in reader:
        # for epoch,acc,loss,val_acc,val_loss in reader:
            losses.append(float(val_loss))
            accuracies.append(float(val_acc))
            top_5_accuracies.append(float(val_top_k_categorical_accuracy))
            cnn_benchmark.append(0.65)  # ridiculous

        plt.plot(accuracies)
        plt.plot(losses)
        plt.plot(top_5_accuracies)
        plt.plot(cnn_benchmark)
        plt.show()

if __name__ == '__main__':
    training_log = 'data/logs/lstm-training-1673733324.0222404.log'
    main(training_log)
