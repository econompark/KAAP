import numpy  as np
import torch
from torch.utils.data import DataLoader, Dataset

def create_data_split_indices(
        dates:                 np.ndarray,
        train_start_date:      str,
        validation_start_date: str,
        test_start_date:       str,
) -> tuple[int, int, int]:
    """
    Create DataLoaders for chronological train, validation, and test periods.

    Parameters
    ----------
    values:
        The complete time series. Its expected shape is either [time] for a
        univariate series or [time, features] for a multivariate series.

    train_start:
        The absolute index where the training period begins.

    validation_start:
        The absolute index where the validation period begins. This index also
        marks the exclusive end of the training period.

    test_start:
        The absolute index where the test period begins. This index also marks
        the exclusive end of the validation period.

    input_length:
        The number of historical observations included in each source sequence.

    forecast_horizon:
        The number of future observations included in each target sequence.

    batch_size:
        The number of sliding-window samples grouped into one batch.

    seed:
        The random seed used to reproduce the training-window shuffle order.

    Notes
    -----
    A DataLoader receives one Dataset instance containing many samples. Each
    sample returned by SlidingWindowDataset has the following structure:

        dataset[j][0]: source sequence of the j-th sample
        dataset[j][1]: target sequence of the j-th sample
        dataset[j][2]: absolute index where the target sequence begins

    The DataLoader repeatedly calls the Dataset's __getitem__ method and stacks
    multiple samples into a batch. For example, if batch_size is 64, the batch
    shapes are typically:

        source batch:       [64, input_length, features]
        target batch:       [64, forecast_horizon, features]
        target-start batch: [64]

    Setting shuffle=True randomly permutes the Dataset sample indices before
    grouping them into batches. It changes both batch composition and batch
    order, but it never changes the chronological order inside an individual
    source or target sequence.

    For example, Dataset indices:

        [0, 1, 2, 3, 4, 5]

    may be shuffled into:

        [4, 1, 5, 0, 2, 3]

    The individual time sequences stored at indices 4, 1, and so on remain
    internally ordered. Only the order in which complete windows are presented
    to the model changes.

    Training data use shuffle=True because each sliding window is treated as an
    independent supervised-learning sample. Validation and test data use
    shuffle=False so predictions remain in chronological order.

    DataLoader Usage
    ----------------
    Calling iter(loader) creates an iterator that tracks the current batch
    position:

        train_iterator = iter(train_loader)

    Calling next(iterator) returns one batch and advances the iterator:

        source_batch, target_batch, start_batch = next(train_iterator)

    Calling next() again on the same iterator returns the following batch:

        next_source, next_target, next_start = next(train_iterator)

    When all batches have been returned, the iterator raises StopIteration.

    A for-loop performs the iter() and next() operations automatically and is
    the standard way to use a DataLoader during training:

        for source_batch, target_batch, start_batch in train_loader:
            ...

    The expression next(iter(train_loader)) is useful for inspecting one batch.
    However, calling it repeatedly creates a new iterator each time instead of
    advancing through consecutive batches.

    Returns
    -------
    train_loader:
        A shuffled DataLoader containing training windows.

    validation_loader:
        A chronological DataLoader containing validation windows.

    test_loader:
        A chronological DataLoader containing test windows.
    """
    dates = np.asarray(
        dates,
        dtype = "datetime64[ns]"
        )

    if dates.ndim != 1:
        raise ValueError(
            "dates must be a one-dimensional array."
        )

    
    if np.any(dates[1:] < dates[:-1]):
        raise ValueError(
            "dates must be sorted in ascending order."
        )

    train_start = int(
        np.searchsorted(
            dates,
            np.datetime64(train_start_date)
        )
    )

    validation_start = int(
        np.searchsorted(
            dates,
            np.datetime64(validation_start_date)
        )
    )

    test_start = int(
        np.searchsorted(
            dates,
            np.datetime64(test_start_date)
        )
    )

    if not (
        train_start
        < validation_start
        < test_start
        < len(dates)
    ):
        raise ValueError(
            "The split dates do not produce valid train,"
            "validation, and test periods."
        )

    return (
        train_start,
        validation_start,
        test_start,
    )

class SlidingWindowDataset(Dataset):
    """
    Return the sliding-window sample at the given dataset index.

    For a SlidingWindowDataset instance named `dataset`:

    - dataset[j][0] is the source sequence of the j-th sample.
    - dataset[j][1] is the target sequence of the j-th sample.
    - dataset[j][2] is the absolute index in the original time series
    where the target sequence of the j-th sample begins.
    """

    def __init__(
            self,
            values:             np.ndarray,
            first_target_index: int,
            end_index:          int,
            input_length:       int,
            forecast_horizon:   int,
    ) -> None:

        values = torch.as_tensor(
            values,
            dtype = torch.float32,
        )

        if values.ndim == 1:
            values = values.unsqueeze(-1)

        if values.ndim != 2:
            raise ValueError(
                "values must have shape [time] or [time, features]."
            )

        if input_length <= 0:
            raise ValueError(
                "input_length must be greater than 0."
            )

        if forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be greater than 0."
            )

        if first_target_index < input_length:
            raise ValueError(
                "There is not enough history before the first target."
            )

        if end_index > len(values):
            raise ValueError(
                "end_index exceeds the number of observations."
            )

        last_target_index = (
            end_index - forecast_horizon
        ) 

        if last_target_index < first_target_index:
            raise ValueError(
                "The selected period is too short."
            )

        self.values           = values
        self.input_length     = input_length
        self.forecast_horizon = forecast_horizon

        # target_starts = (train_start + input_length) ~ (validation_start - forecast_horizon)
        # i.e. target_starts = after input ~ before validation
        self.target_starts = torch.arange(
            first_target_index,
            last_target_index + 1,
            dtype = torch.long,
        )
    
    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(
            self,
            index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        target_start = self.target_starts[index]

        source_start = (
            target_start - self.input_length
        )

        # source = values of "target_start - input_length ~ target_start[index] - 1"
        source = self.values[
            source_start:target_start
        ]

        # target = values of "target_start[index] ~ target_starts[index] + forecasati_horizon - 1"
        target = self.values[
            target_start:target_start + self.forecast_horizon
        ]

        return (
            source,
            target,
            target_start,
        )

def create_data_loaders(
        values:           np.ndarray,
        train_start:      int,
        validation_start: int,
        test_start:       int,
        input_length:     int,
        forecast_horizon: int,
        batch_size:       int,
        seed:             int = 42,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:

    train_dataset = SlidingWindowDataset(
        values             = values,
        first_target_index = train_start + input_length,
        end_index          = validation_start,
        input_length       = input_length,
        forecast_horizon   = forecast_horizon,
    )

    validation_dataset = SlidingWindowDataset(
        values             = values,
        first_target_index = validation_start,
        end_index          = test_start,
        input_length       = input_length,
        forecast_horizon   = forecast_horizon,
    )

    test_dataset = SlidingWindowDataset(
        values             = values,
        first_target_index = test_start,
        end_index          = len(values),
        input_length       = input_length,
        forecast_horizon   = forecast_horizon,
    )

    # random number generator for shuffle
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle    = True,
        generator  = generator
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size = batch_size,
        shuffle    = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle    = False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )