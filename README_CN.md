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
* 下载 [ShapeNet dataset v1](https://www.shapenet.org/) 保存在 `data/external/ShapeNet`
* 从 Choy et al. 2016 下载 [renderings and voxelizations](http://3d-r2n2.stanford.edu/)  并且展开到 `data/external/Choy2016` 
* 按照下述的指导在 `external/mesh-fusion` 文件夹中构建我们修改过的 [mesh-fusion](https://github.com/davidstutz/mesh-fusion)

现在，构建数据集：
```
cd scripts
bash dataset_shapenet/build.sh
```

这个命令将在文件夹 `data/ShapeNet.build` 构建数据集，执行下面的命令安装数据集：
```
bash dataset_shapenet/install.sh
```

如果一切都这，数据集将会拷贝到文件夹 `data/ShapeNet`.

## 使用
当你已经安装了所有的二进制依赖，也获取了预处理数据，则可以运行预训练模型，并且从头开始训练新的模型

### 生成
为了使用训练好的模型生成网格可以执行下面的命令：
```
python generate.py CONFIG.yaml
```
可以使用其他正确的配置文件来替换 `CONFIG.yaml`

最简单的方式是使用预训练模型。你可以下面的配置文件中的一个文件：
```
configs/img/onet_pretrained.yaml
configs/pointcloud/onet_pretrained.yaml
configs/voxels/onet_pretrained.yaml
configs/unconditional/onet_cars_pretrained.yaml
configs/unconditional/onet_airplanes_pretrained.yaml
configs/unconditional/onet_sofas_pretrained.yaml
configs/unconditional/onet_chairs_pretrained.yaml
```
结果对应着论文中的实验。我们的脚本将会自动下载模型的检查点，并且运行生成。输出在 `out/*/*/pretrained`  文件夹中

请注意：配置文件`*_pretrained.yaml` 仅仅用于生成，而不能用于训练新的模型：当这些配置文件用于训练时，模型将会从头开始训练，但是当推理我们的代码时仍然使用预训练模型

### 评估
对于模型的评估，我们提供两个脚本：`eval.py` 和 `eval_meshes.py`.

主要评估脚本 `eval_meshes.py` 的运行方式：

```
python eval_meshes.py CONFIG.yaml
```
脚本中在前一个步骤生成网格，然后使用一个标准的协议评估它们。使用 [pandas](https://pandas.pydata.org/) 处理并输出 `.pkl`/`.csv` 文件在对应的生成文件夹中。

对于一个快速评估，可以使用下面脚本：
```
python eval.py CONFIG.yaml
```
这个脚本运行特定评估的快速方式，从而获取一些基本的参量，这些参量不需要提取网格就可以方便地计算。这个评估在训练期间也可以在验证集上自动进行。

论文中报告的所有结果都可以使用 `eval_meshes.py` 脚本获得

### 训练
最后，为了从头开始训练一个新的网络，执行：
```
python train.py CONFIG.yaml
```
其中的 `CONFIG.yaml` 可以使用你想要使用的配置文件名称替换。

你可以使用 [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) 访问 <http://localhost:6006> 监视训练过程：
```
cd OUTPUT_DIR
tensorboard --logdir ./logs --port 6006
```
可以使用不同的输入文件夹替换 `OUTPUT_DIR` 

对于有效的训练选择，可以参考`configs/default.yaml`.

# 注释
* 本文中使用随机剪裁和缩放来增强输入图像数据集。然而，这种图像数据集增强技术降低了 ShapeNet 测试集的性能。预训练模型使用 `configs/img/onet_pretrained.yaml` 载入，因此不需要数据集增强进行训练，并且相比论文中的模型拥有更好的性能。这个更新表如下：
  ![Updated table for single view 3D reconstruction experiment](img/table_img2mesh.png)为了完整性，项目还为论文中使用的模型提供了训练后的权重 `configs/img/onet_legacy_pretrained.yaml`.
* 请注意：本文模型和基线是在*水密模型*上完成训练与评估，但是归一化到单位立方体是在*非水密网格*（为了与 Choy et al. 中的体素化保持一致）上完成的。因此，采样点云的包围盒通常比单位立方体稍大一点，并且与原始的 ShapeNet 网格采样的点云稍有不同。

# 进一步的信息
请检查下述的同期论文，它们拥有相似的理念：
* [Park et al. - DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (2019)](https://arxiv.org/abs/1901.05103)
* [Chen et al. - Learning Implicit Fields for Generative Shape Modeling (2019)](https://arxiv.org/abs/1812.02822)
* [Michalkiewicz et al. - Deep Level Sets: Implicit Surface Representations for 3D Shape Inference (2019)](https://arxiv.org/abs/1901.06802)

