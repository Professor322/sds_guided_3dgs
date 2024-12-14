# 3D Gaussian splatting guided by Score Distillation Sampling
3D Gaussian splatting guided by score distillation sampling <br>

To implement the Gaussian Splatting part we decided utilize [gsplat library](https://github.com/nerfstudio-project/gsplat) <br>
Arxiv reference: https://arxiv.org/abs/2409.06765

## Gsplat installation notes
* I decided to work on the latest mainline as contributors pushing their code to fix existing library
* On the mainline maintainers introduced some compression techniques. Not from the dicsussed paper, but rather just a meaningfull way of doing .png compression.
* Working with their mainline necessitates install it from the source using `pip install git+https://github.com/nerfstudio-project/gsplat.git `. <br>
  During installation library will compile cuda kernels. If you hit any problems running out of CPU RAM try reducing `MAX_JOBS`.

## Notes on working with gsplat
* As a base implementation I took code from `examples` directory and got rid of 
* Tried to use gsplat to render scenes from NerF 360 data set - real time rendering works along with visualisation, checkpointing, validation

## Some thoughts
* It seems like in the world of 3d rendering people are chasing training speed, rendering speed and quality. SDS will definetly decrease training speed.
* If all goes well I think the code from this repo should be pushed as a pull request to gsplat library. 
* Beauty of 3D Gaussian splatting is that is start with small number of ellipsoids, so that not all GPU memory allocated in the beginning of the training <br>
  This allows to prototype on the low end PCs


## Some useful commands

To download MipNerf dataset. (It will take some time)
```
python ./datasets/download_dataset.py
```

To create point cloud from the existing data (courtesy of Google NeRF360). Make sure that `data_path` has directory `images`
```
./scripts/local_colmap_and_resize.sh <data_path>
```

To check how well loading colmap loading works. This will create a video clip in results directory with <br>
points mapped to train images
```
python3 ./datasets/colmap.py --data_dir=data/360_v2/bicycle
```

To train 
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --eval_steps -1 --disable_viewer --data_factor 2 \
    --render_traj_path "ellipse" \
    --data_dir <data_dir> \
    --result_dir results/benchmark/<scene>
```

To render from checkpoint
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable_viewer --data_factor $DATA_FACTOR \
    --render_traj_path "ellipse" \
    --data_dir <data_dir> \
    --result_dir results/benchmark/<scene> \
    --ckpt results/benchmark/<scene>/<checkpoint>
```

## TODO
* Missing SDS integation
* Check how SSIM works
