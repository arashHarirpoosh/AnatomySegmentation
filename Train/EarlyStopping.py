class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        else:
            if (self.best_loss - loss) > self.min_delta:
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
                print(f'Early Stopping patience: {self.counter}, best loss: {self.best_loss}, current_loss: {loss}')
                if self.counter >= self.tolerance:
                    self.early_stop = True
