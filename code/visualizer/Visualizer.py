import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue
from threading import Thread
import os
import time


class Visualizer:

    def __init__(self, options):
        self.options = options

        # used by main and calculation thread
        self.data_queue = queue.Queue()
        self.finished = False

        # only used by main thread (all list of train/test have to have the same length)
        self.train_loss     = []
        self.train_accuracy = []
        self.train_progress = []

        self.test_loss     = []
        self.test_accuracy = []
        self.test_progress = []

        # creating figure
        w, h = plt.figaspect(0.5)
        self.fig = plt.figure(figsize=(w, h))
        self.fig.suptitle('Hand Gesture Recognition', fontsize=18, y=0.96)
        # self.fig.canvas.mpl_connect('close_event', lambda evt: os._exit(0))

        self.axLoss = self.fig.add_subplot(1, 2, 1)
        self.lineTrainLoss, = self.axLoss.plot([], [], 'b-')
        self.lineTestLoss,  = self.axLoss.plot([], [], 'r-')
        self.axLoss.set_ylim(0, 1)
        self.axLoss.set_xlim(0, 1)
        self.axLoss.legend(['Train Loss', 'Test Loss'], loc="upper right")

        self.axAcc = self.fig.add_subplot(1, 2, 2)
        self.lineTrainAcc, = self.axAcc.plot([], [], 'b-')
        self.lineTestAcc, = self.axAcc.plot([], [], 'r-')
        self.axAcc.set_ylim(0, 1)
        self.axAcc.set_xlim(0, 1)
        self.axAcc.legend(['Train Accuracy', 'Test Accuracy'], loc="lower right")

        self.lossLim     = 4
        self.progressLim = 5

    def _updateAnimation(self, frame):

        # fetch new data from the calculation thread
        while not self.data_queue.empty():
            (isTest, loss, accuracy, progress) = self.data_queue.get_nowait()

            if isTest:
                self.test_loss.append(loss)
                self.test_accuracy.append(accuracy)
                self.test_progress.append(progress)
            else:
                self.train_loss.append(loss)
                self.train_accuracy.append(accuracy)
                self.train_progress.append(progress)

            self.lossLim = max(self.lossLim, loss * 1.05)
            self.progressLim = max(self.progressLim, progress * 1.05)

        self.lineTrainLoss.set_data(self.train_progress, self.train_loss)
        self.lineTrainAcc.set_data(self.train_progress, self.train_accuracy)
        self.lineTestLoss.set_data(self.test_progress, self.test_loss)
        self.lineTestAcc.set_data(self.test_progress, self.test_accuracy)

        self.axLoss.set_xlim(0, self.progressLim)
        self.axLoss.set_ylim(0, self.lossLim)
        self.axAcc.set_xlim(0, self.progressLim)

        if self.finished:
            self.fig.savefig(self.options.output_path(time.strftime("%Y%m%d-%H%M%S") + ".png"))
            self.finished = False

        return self.axLoss, self.axAcc,

    def show(self, resultGenerator):
        self.finished = False

        if self.options.interactiveGUI:
            def main():
                for isTest, loss, accuracy, progress in resultGenerator():
                    self.data_queue.put((isTest, loss, accuracy, progress))

                self.finished = True

            calc_thread = Thread(target=main, args=())
            calc_thread.start()

            ani = animation.FuncAnimation(self.fig, func=self._updateAnimation, frames=None, init_func=None, blit=False, interval=10, repeat=False)
            plt.show()

        else:
            for isTest, loss, accuracy, progress in resultGenerator():
                self.data_queue.put((isTest, loss, accuracy, progress))

            self.finished = True

            self._updateAnimation(0)
            plt.show()
