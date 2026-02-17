import tensorflow_datasets as tfds
import numpy as np

builder_full = tfds.builder("dsynth_atomic_tasks:1.3.0",
                            data_dir="/home/jovyan/shares/SR006.nfs2/tensorflow_datasets")
ds_full = builder_full.as_dataset(split='all')
ds_full_iterator = iter(ds_full)

for idx, episode in enumerate(ds_full_iterator):
    if idx >= 5:
        break

    for step_idx, step in enumerate(episode['steps']):
        action = step['action'].numpy()
        print(f"Episode {idx}, Step {step_idx}:")
        print(f"  Action shape: {action.shape}")
        print(f"  Action dtype: {action.dtype}")
        print(f"  Action values: {action}")

        if step_idx >= 2:
            break
