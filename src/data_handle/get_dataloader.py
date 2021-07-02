from torch.utils.data import DataLoader

def get_dataloader(split, batch_size, num_workers, shuffle, dataset_cfg):
    if "JRDB" in dataset_cfg["data_dir"]:
        from .jrdb_dataset import JRDBBoxRegressionDataset
        ds = JRDBBoxRegressionDataset(split, dataset_cfg)
    else:
        raise RuntimeError(f"Unknown dataset {dataset_cfg['name']}.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=ds.collate_batch
    )
