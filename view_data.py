import grain
import pdb
import numpy as np
import pickle
from PIL import Image
import imageio

# grain_source_mc = grain.sources.ArrayRecordDataSource(["/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered/train/6.9_Player8-f153ac423f61-20210529-190444_chunk001.array_record"])
grain_source_mc = grain.sources.ArrayRecordDataSource(["/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered/val/10.0_woozy-ruby-ostrich-36e6a60ea6e6-20220420-154253_chunk001.array_record"])
# grain_source_cr = grain.sources.ArrayRecordDataSource([
#     '/home/4bkang/rl/jasmine/data/data/coinrun_episodes/val/data_0000.array_record',
#     '/home/4bkang/rl/jasmine/data/data/coinrun_episodes/val/data_0001.array_record'
# ])

element = pickle.loads(grain_source_mc[0])
# element_cr = pickle.loads(grain_source_cr[0])

# pdb.set_trace()

episode_tensor = np.frombuffer(element["raw_video"], dtype=np.uint8)
episode_tensor = episode_tensor.reshape(element["sequence_length"], 90, 160, 3)
# episode_tensor = episode_tensor.reshape(element["sequence_length"], 64, 64, 3)

# Save as MP4 video
output_path = "/home/4bkang/rl/jasmine/episode.mp4"
imageio.mimwrite(output_path, episode_tensor, fps=10, codec='libx264')
print(f"Saved video to {output_path}")

