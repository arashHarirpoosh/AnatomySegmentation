class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0, min_lr=1e-6, mode='min'):
        self.mode = mode
        self.counter = 0
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss, lr):
        if self.best_loss is None:
            self.best_loss = loss
        else:
            if self.mode == 'min':
                if (self.best_loss - loss) > self.min_delta:
                    self.best_loss = loss
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'Early Stopping patience: {self.counter}, best loss: {self.best_loss}, current_loss: {loss}')
                    if self.counter >= self.tolerance:
                        self.early_stop = True
            else:
                if (loss - self.best_loss) > self.min_delta:
                    self.best_loss = loss
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'Early Stopping patience: {self.counter}, best loss: {self.best_loss}, current_loss: {loss}')
                    if self.counter >= self.tolerance:
                        self.early_stop = True
            if lr > self.min_lr and self.early_stop:
                print(f'Maximum patience is ended, but more time is needed (lr > {self.min_lr})')
                self.early_stop = False
