import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class DistributedModalitySampler(Sampler):
    """Yield an exact per-rank modality mix for every training batch."""

    def __init__(
            self, dataset, modality_batch_counts, num_replicas=1, rank=0,
            shuffle=True, seed=0):
        if rank < 0 or rank >= num_replicas:
            raise ValueError('rank must be in [0, num_replicas).')
        self.dataset = dataset
        self.modality_batch_counts = {
            str(group): int(count)
            for group, count in modality_batch_counts.items()
        }
        if not self.modality_batch_counts or any(
                count <= 0 for count in self.modality_batch_counts.values()):
            raise ValueError('All modality batch counts must be positive.')
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.local_batch_size = sum(self.modality_batch_counts.values())
        self.global_batch_size = self.local_batch_size * self.num_replicas
        self.num_batches = int(math.ceil(len(dataset) / self.global_batch_size))

        grouped_indices = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            grouped_indices[str(sample['group'])].append(index)
        missing = set(self.modality_batch_counts) - set(grouped_indices)
        if missing:
            raise ValueError('Dataset has no samples for: {}'.format(sorted(missing)))
        self.grouped_indices = dict(grouped_indices)

    @staticmethod
    def _expanded_indices(indices, total, rng, shuffle):
        expanded = []
        while len(expanded) < total:
            cycle = list(indices)
            if shuffle:
                rng.shuffle(cycle)
            expanded.extend(cycle)
        return expanded[:total]

    def __iter__(self):
        group_streams = {}
        for offset, (group, local_count) in enumerate(
                sorted(self.modality_batch_counts.items())):
            total = self.num_batches * local_count * self.num_replicas
            rng = random.Random(self.seed + self.epoch * 1009 + offset)
            group_streams[group] = self._expanded_indices(
                self.grouped_indices[group], total, rng, self.shuffle
            )

        rank_indices = []
        for batch_index in range(self.num_batches):
            local_batch = []
            for group, local_count in sorted(self.modality_batch_counts.items()):
                global_count = local_count * self.num_replicas
                start = batch_index * global_count + self.rank * local_count
                local_batch.extend(group_streams[group][start:start + local_count])
            if self.shuffle:
                batch_rng = random.Random(
                    self.seed + self.epoch * 1000003 + batch_index * 97 + self.rank
                )
                batch_rng.shuffle(local_batch)
            rank_indices.extend(local_batch)
        return iter(rank_indices)

    def __len__(self):
        return self.num_batches * self.local_batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
