from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data

from .batch import Batch
from .sequence import Sequence

dataset_dir = Path(__file__).parents[3] / "data"


def list_datasets(folder: Union[str, Path] = dataset_dir):
    """List all datasets in the data folder."""
    files = sorted(file.stem for file in Path(folder).iterdir() if file.suffix == ".pt")
    dirs = sorted(d for d in Path(folder).iterdir() if d.is_dir() and not d.name.startswith("_"))
    for d in dirs:
        if d.is_dir() and not d.stem.startswith("_"):
            for f in d.iterdir():
                if f.suffix == ".pt":
                    files.append(d.stem + "/" + f.stem)
    return files


def load_dataset(name: str, folder: Union[str, Path] = dataset_dir,) -> "SequenceDataset":
    """Load a dataset from the given data folder."""
    if not name.endswith(".pt"):
        name += ".pt"
    path_to_file = Path(folder) / name
    return SequenceDataset.from_file(path_to_file)


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset of variable-length event sequences."""

    def __init__(
        self,
        sequences: List[Sequence],
        num_marks: int = 0,
        metadata: Optional[dict] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.num_marks = num_marks
        self.metadata = metadata
        self.transform = transform
        self.pre_transform = pre_transform

        if pre_transform is not None:
            sequences = [pre_transform(seq) for seq in sequences]

        self.sequences = sequences

    def set_transform(
        self,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.pre_transform = pre_transform

        if pre_transform is not None:
            self.sequences = [pre_transform(seq) for seq in self.sequences]

    @staticmethod
    def from_file(path_to_file: Union[str, Path]) -> "SequenceDataset":
        """Load the dataset from a file."""
        dataset = torch.load(path_to_file)
        sequences = [Sequence(**seq) for seq in dataset["sequences"]]
        return SequenceDataset(
            sequences, num_marks=dataset["num_marks"], metadata=dataset.get("metadata")
        )

    def to_file(self, path_to_file: Union[str, Path]):
        """Save the dataset to a file."""
        path_to_file = str(path_to_file)
        if not path_to_file.endswith(".pt"):
            path_to_file += ".pt"
        data_dict = {
            "sequences": [seq.to_dict() for seq in self.sequences],
            "num_marks": self.num_marks,
            "metadata": self.metadata,
        }
        torch.save(data_dict, path_to_file)

    def __getitem__(self, key: int) -> Sequence:
        item = self.sequences[key]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __add__(self, other: "SequenceDataset") -> "SequenceDataset":
        if not isinstance(other, SequenceDataset):
            raise ValueError(f"other must be a SequenceDataset (got {type(other)})")
        new_num_marks = max(self.num_marks, other.num_marks)
        new_sequences = self.sequences + other.sequences
        return SequenceDataset(new_sequences, num_marks=new_num_marks)

    @property
    def num_events(self):
        """Total number of events in all sequences."""
        return sum(len(seq) for seq in self.sequences)

    def get_dataloader(
        self, batch_size: int = 32, shuffle: bool = True, **kwargs
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=Batch.from_list, **kwargs
        )

    def train_val_test_split(
        self, train_size=0.6, val_size=0.2, test_size=0.2, seed=None, shuffle=True,
    ) -> Tuple["SequenceDataset", "SequenceDataset", "SequenceDataset"]:
        """Split the sequences into train, validation and test subsets."""
        if train_size < 0 or val_size < 0 or test_size < 0:
            raise ValueError("train_size, val_size and test_size must be >= 0.")
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size, val_size and test_size must add up to 1.")

        if seed is not None:
            np.random.seed(seed)

        all_idx = np.arange(len(self))
        if shuffle:
            np.random.shuffle(all_idx)

        train_end = int(train_size * len(self))  # idx of the last train sequence
        val_end = int((train_size + val_size) * len(self))  # idx of the last val seq

        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]

        train_sequences = [self.sequences[idx] for idx in train_idx]
        val_sequences = [self.sequences[idx] for idx in val_idx]
        test_sequences = [self.sequences[idx] for idx in test_idx]

        return (
            SequenceDataset(train_sequences, num_marks=self.num_marks),
            SequenceDataset(val_sequences, num_marks=self.num_marks),
            SequenceDataset(test_sequences, num_marks=self.num_marks),
        )

    def to(self, device):
        for seq in self.sequences:
            seq.to(device)
