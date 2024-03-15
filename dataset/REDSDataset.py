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

    def __init__(
        self,
        gt_dir,
        lq_dir,
        scale_factor,
        patch_size,
        num_input_frames,
        filename_tmpl,
        max_keys,
        val_partition,
        is_test,
    ):
        self.gt_dir = gt_dir
        self.lq_dir = lq_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_test = is_test
        self.num_input_frames = num_input_frames
        self.filename_tmpl = filename_tmpl
        self.max_keys = max_keys
        self.val_partition = val_partition

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
        #TODO 重写一下，速度太慢
        gt_sequence, lq_sequence = generate_segment_indices(
            self.gt_seq_paths[idx],
            self.lq_seq_paths[idx],
            num_input_frames=self.num_input_frames,
            filename_tmpl=self.filename_tmpl,
        )
        if not self.is_test:
            gt_sequence, lq_sequence = self.transform(gt_sequence, lq_sequence)
        return gt_sequence, lq_sequence
