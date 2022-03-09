# 占位网络
![Example 1](img/00.gif)
![Example 2](img/01.gif) 
![Example 3](img/02.gif)

仓库中的代码再现了论文[Occupancy Networks - Learning 3D Reconstruction in Function Space](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks)中的结果。

本文中可以了解使用指导的细节用于训练你自己的模型和使用下面的预训练模型。

如果你觉得我们的代码和论文有用，请考虑引用：


```tex
@inproceedings{Occupancy Networks,
    title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
    author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```

## 安装
首先，你必须确保所有的依赖都已经安装。最简单的方式就是使用 [anaconda](https://www.anaconda.com/) 来实现。

你可以创建一个 anaconda 环境，称作：`mesh_funcspace`：
```
conda env create -f environment.yaml
conda activate mesh_funcspace
```

接下来，编译扩展的模型，你可以如下操作：
```
python setup.py build_ext --inplace
```

为了编译 dmc(Deep Marching Cubes) 扩展，必须拥有支持 CUDA 设备的配置。如果你遇到任何错误，你可以简单的注释掉 `setup.py` 中的 `dmc_*` 依赖，同时还需要注释掉 `im2mesh/config.py` 中的 `dmc` 导入。

## 示范
![Example Input](img/example_input.png)
![Example Output](img/example_output.gif)

现在，你可以使用 `demo` 文件夹中提供的输入图像测试我们的代码。方式如下：

```
python generate.py configs/demo.yaml
```
这个脚本会创建一个文件夹 `demo/generation` 用于保存输出的网格。脚本会拷贝输入到 `demo/generation/inputs` 文件夹中，创建网格在 `demo/generation/meshes` 文件夹中。而且，脚本创建 `demo/generation/vis` 文件夹用于将输入和输出拷贝到一起。

## 数据集

为了评估预训练模型或者从头训练一个新的模型，可以获取数据集。下面有两种获取方式：

1. 下载预处理好的数据
2. 下载 ShapeNet 数据集，执行预处理管线

请注意：执行预处理管线时需要大量的时间和硬盘空间。除非你需要应用我们的方法到新的数据集上，否则我们建议你使用第一个选择。

### 预处理数据
预处理数据 (73.4 GB) 的下载：

```
bash scripts/download_data.sh
```

脚本将会自动下载和展开到 `data/ShapeNet` 文件夹

### 构建数据集
自己预处理数据集可以执行下面的步骤：
* download the [ShapeNet dataset v1](https://www.shapenet.org/) and put into `data/external/ShapeNet`. 
* download the [renderings and voxelizations](http://3d-r2n2.stanford.edu/) from Choy et al. 2016 and unpack them in `data/external/Choy2016` 
* build our modified version of [mesh-fusion](https://github.com/davidstutz/mesh-fusion) by following the instructions in the `external/mesh-fusion` folder

You are now ready to build the dataset:
```
cd scripts
bash dataset_shapenet/build.sh
```

This command will build the dataset in `data/ShapeNet.build`.
To install the dataset, run
```
bash dataset_shapenet/install.sh
```

If everything worked out, this will copy the dataset into `data/ShapeNet`.

## Usage
When you have installed all binary dependencies and obtained the preprocessed data, you are ready to run our pretrained models and train new models from scratch.

### Generation
To generate meshes using a trained model, use
```
python generate.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.

The easiest way is to use a pretrained model.
You can do this by using one of the config files
```
configs/img/onet_pretrained.yaml
configs/pointcloud/onet_pretrained.yaml
configs/voxels/onet_pretrained.yaml
configs/unconditional/onet_cars_pretrained.yaml
configs/unconditional/onet_airplanes_pretrained.yaml
configs/unconditional/onet_sofas_pretrained.yaml
configs/unconditional/onet_chairs_pretrained.yaml
```
which correspond to the experiments presented in the paper.
Our script will automatically download the model checkpoints and run the generation.
You can find the outputs in the `out/*/*/pretrained` folders.

Please note that the config files  `*_pretrained.yaml` are only for generation, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pretrained model.

### Evaluation
For evaluation of the models, we provide two scripts: `eval.py` and `eval_meshes.py`.

The main evaluation script is `eval_meshes.py`.
You can run it using
```
python eval_meshes.py CONFIG.yaml
```
The script takes the meshes generated in the previous step and evaluates them using a standardized protocol.
The output will be written to `.pkl`/`.csv` files in the corresponding generation folder which can be processed using [pandas](https://pandas.pydata.org/).

For a quick evaluation, you can also run
```
python eval.py CONFIG.yaml
```
This script will run a fast method specific evaluation to obtain some basic quantities that can be easily computed without extracting the meshes.
This evaluation will also be conducted automatically on the validation set during training.

All results reported in the paper were obtained using the `eval_meshes.py` script.

### Training
Finally, to train a new network from scratch, run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs --port 6006
```
where you replace `OUTPUT_DIR` with the respective output directory.

For available training options, please take a look at `configs/default.yaml`.

# Notes
* In our paper we used random crops and scaling to augment the input images. 
  However, we later found that this image augmentation decreases performance on the ShapeNet test set.
  The pretrained model that is loaded in `configs/img/onet_pretrained.yaml` was hence trained without data augmentation and has slightly better performance than the model from the paper. The updated table looks a follows:
  ![Updated table for single view 3D reconstruction experiment](img/table_img2mesh.png)
  For completeness, we also provide the trained weights for the model which was used in the paper in  `configs/img/onet_legacy_pretrained.yaml`.
* Note that training and evaluation of both our model and the baselines is performed with respect to the *watertight models*, but that normalization into the unit cube is performed with respect to the *non-watertight meshes* (to be consistent with the voxelizations from Choy et al.). As a result, the bounding box of the sampled point cloud is usually slightly bigger than the unit cube and may differ a little bit from a point cloud that was sampled from the original ShapeNet mesh.

# Futher Information
Please also check out the following concurrent papers that have proposed similar ideas:
* [Park et al. - DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (2019)](https://arxiv.org/abs/1901.05103)
* [Chen et al. - Learning Implicit Fields for Generative Shape Modeling (2019)](https://arxiv.org/abs/1812.02822)
* [Michalkiewicz et al. - Deep Level Sets: Implicit Surface Representations for 3D Shape Inference (2019)](https://arxiv.org/abs/1901.06802)

