class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Early stops the training if validation loss doesn't improve for a given
        number of consecutive epochs.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each epoch when early stopping is triggered.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Updates the early stopping state based on the validation loss.
        Args:
            val_loss (float): Validation loss.
        Returns:
            True if the training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif self.best_score - val_loss < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping triggered after {self.counter} epochs.')
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop

class EarlyCorrelationStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Early stops the training if correlation doesn't improve for a given
        number of consecutive epochs.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each epoch when early stopping is triggered.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_corr):
        """
        Updates the early stopping state based on the validation correlation.
        Args:
            val_corr (float): Validation correlation.
        Returns:
            True if the training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_corr
        elif val_corr - self.best_score <= self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping triggered after {self.counter} epochs.')
                self.early_stop = True
        else:
            self.best_score = val_corr
            self.counter = 0
        return self.early_stop

class EarlyLossCorrelationStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Early stops the training if validation loss doesn't improve for a given
        number of consecutive epochs.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each epoch when early stopping is triggered.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_val = None
        self.best_corr = None
        self.early_stop = False

    def __call__(self, val_loss, val_corr):
        """
        Updates the early stopping state based on the validation loss.
        Args:
            val_loss (float): Validation loss.
            val_corr (float): Validation correlation
        Returns:
            True if the training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_corr = val_corr
        elif (val_loss > self.best_score - self.delta) and (val_corr < self.best_corr - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping triggered after {self.counter} epochs.')
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop

