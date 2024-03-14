"""
This code is based on Open-MMLab's one.
https://github.com/open-mmlab/mmediting
"""

from torch.utils.data import Dataset
import os

from .utils import (
    generate_segment_indices,
    pair_random_crop_seq,
    pair_random_transposeHW_seq,
    pair_random_flip_seq,
)


class REDSDataset(Dataset):
    """REDS dataset for video super resolution."""

    def __init__(self, args, is_test=False):
        self.gt_dir = args.gt_dir
        self.lq_dir = args.lq_dir
        self.scale_factor = args.scale_factor
        self.patch_size = args.patch_size
        self.is_test = is_test
        self.num_input_frames = args.num_input_frames if not self.is_test else 100
        self.filename_tmpl = args.filename_tmpl
        self.max_keys = args.max_keys
        self.val_partition = args.val_partition

        self.keys = self.load_keys()
        self.gt_seq_paths = [os.path.join(self.gt_dir, k) for k in self.keys]
        self.lq_seq_paths = [os.path.join(self.lq_dir, k) for k in self.keys]

    def load_keys(self):
        keys = [f"{i:03d}" for i in range(0, self.max_keys)]

        if self.val_partition == "REDS4":
            val_partition = ["000", "011", "015", "020"]
        elif self.val_partition == "official":
            val_partition = [f"{i:03d}" for i in range(240, 270)]
        else:
            raise ValueError(
                f"Wrong validation partition {self.val_partition}."
                f'Supported ones are ["official", "REDS4"]'
            )

        if self.is_test:
            keys = [v for v in keys if v in val_partition]
        else:
            keys = [v for v in keys if v not in val_partition]

        return keys

    def transform(self, gt_seq, lq_seq):
        gt_transformed, lq_transformed = pair_random_crop_seq(
            gt_seq, lq_seq, patch_size=self.patch_size, scale_factor=self.scale_factor
        )
        gt_transformed, lq_transformed = pair_random_flip_seq(
            gt_transformed, lq_transformed, p=0.5
        )
        gt_transformed, lq_transformed = pair_random_transposeHW_seq(
            gt_transformed, lq_transformed, p=0.5
        )
        return gt_transformed, lq_transformed

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        gt_sequence, lq_sequence = generate_segment_indices(
            self.gt_seq_paths[idx],
            self.lq_seq_paths[idx],
            num_input_frames=self.num_input_frames,
            filename_tmpl=self.filename_tmpl,
        )
        if not self.is_test:
            gt_sequence, lq_sequence = self.transform(gt_sequence, lq_sequence)
        return gt_sequence, lq_sequence
