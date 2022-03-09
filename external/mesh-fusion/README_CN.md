# 通过 “TSDF 融合” 得到水密和精简的网格

[这个仓库]()包含一个简化的 Python 管线用于从任意三角形网格（文件格式为：`.off`）中获取水密的和精简的网格。方法主要基于 Gernot Riegler 的 [pyrender](https://github.com/griegler/pyrender)  和  [pyfusion](https://github.com/griegler/pyfusion) 的自适应版本：还使用了 [PyMCubes](https://github.com/pmneila/PyMCubes)

如果你使用了这个代码，请引用下述的工作：

    @article{Stutz2018ARXIV,
        author    = {David Stutz and Andreas Geiger},
        title     = {Learning 3D Shape Completion under Weak Supervision},
        journal   = {CoRR},
        volume    = {abs/1805.07290},
        year      = {2018},
        url       = {http://arxiv.org/abs/1805.07290},
    }

参考上述的 GitHub 仓库添加附加的引用，也可以查询对应的[项目主页](http://davidstutz.de/projects/shape-completion/).

管线包括三个步骤：

1. 使用`1_scale.py`缩放，缩放比例为`[-0.5,0.5]^3` ，附带可靠的填充
2. 使用`2_fusion.py`呈现和融合，呈现第一步中每个网格的视图，并且使用它们执行 TSDF 融合
3. 使用 MeshLab 执行精简

下面描述的是结果；请注意形状变厚和坐标轴的改变（由于“移动立方体算法”）

![Example of watertight and simplified mesh.](screenshot.jpg?raw=true "Example of watertight and simplified mesh.")

## 安装

如果给定了 Python 的工作安装，管线大部分流程是自满足的。而且，它需要 [MeshLab](http://www.meshlab.net/) 完成精简

使用的 [pyfusion](https://github.com/griegler/pyfusion) 库需要 CMake 和 Cython；可选的是使用 CUDA 或者 OpenMP 提高效率。[pyrender](https://github.com/griegler/pyrender) 也需要 Cython，还需要 OpenGL, GLUT 和 GLEW，参见 `librender/setup.py` 了解细节。[PyMCubes](https://github.com/pmneila/PyMCubes) 需要 Cython。这三个库都被包含在这个仓库中。

构建过程如下（GPU 版本）：

    # build pyfusion
    # use libfusioncpu alternatively!
    cd libfusiongpu
    mkdir build
    cd build
    cmake ..
    make
    cd ..
    python setup.py build_ext --inplace
    
    cd ..
    # build pyrender
    cd librender
    python setup.py build_ext --inplace
    
    cd ..
    # build PyMCubes
    cd libmcubes
    python setup.py build_ext --inplace

## 使用

在 `examples/0_in` 中附带的例子（采集于 [ModelNet](http://modelnet.cs.princeton.edu/)）说明了用法

第一步，使用下面的代码缩放：

    python 1_scale.py --in_dir=examples/0_in/ --out_dir=examples/1_scaled/

现在，每个模型可以使用100个视图（球体上均匀采样）去呈现：

    2_fusion.py --mode=render --in_dir=examples/1_scaled/ --depth_dir=examples/2_depth/ --out_dir=examples/2_watertight/

呈现的细节可以使用下面的选项进行控制：

    --n_views N_VIEWS     Number of views per model.
    --image_height IMAGE_HEIGHT
                          Depth image height.
    --image_width IMAGE_WIDTH
                          Depth image width.
    --focal_length_x FOCAL_LENGTH_X
                          Focal length in x direction.
    --focal_length_y FOCAL_LENGTH_Y
                          Focal length in y direction.
    --principal_point_x PRINCIPAL_POINT_X
                          Principal point location in x direction.
    --principal_point_y PRINCIPAL_POINT_Y
                          Principal point location in y direction.
    --depth_offset_factor DEPTH_OFFSET_FACTOR
                          The depth maps are offsetted using
                          depth_offset_factor*voxel_size.
    --resolution RESOLUTION
                          Resolution for fusion.
    --truncation_factor TRUNCATION_FACTOR
                          Truncation for fusion is derived as
                          truncation_factor*voxel_size.

呈现期间，一个小的偏移被加入到深度图中。这个偏移对于具有稀疏细节的网格非常重要，例如：提供的椅子。本质上 ，这个是结构的厚度。在代码中，偏移量的计算如下：

```python
voxel_size = 1/resolution
offset = depth_offset_factor*voxel_size
```

现在，融合可以使用下面的命令执行：

    python 2_fusion.py --mode=fuse --in_dir=examples/1_scaled/ --depth_dir=examples/2_depth/ --out_dir=examples/2_watertight/

对于融合，分辨率与截断因子非常重要。在实践中，截断因子在`[0, ..., 15]`之间；然后，截断阈值的计算如下：

    voxel_size = 1/resolution
    truncation = truncatioN_factor*voxel_size

**请注意：渲染与融合被分开是因为渲染不能工作在所有的机器上，特别是远程连接的（如：ssh连接）没有显示器的机器上**

最后，精简使用`meshlabserver`完成，确认软件已经被安装：

    python 3_1_simplify.py --in_dir=examples/2_watertight/ --out_dir=examples/3_out/

所有步骤的运行结果在上述屏幕截图中描述。

## License

License for source code corresponding to:

D. Stutz, A. Geiger. **Learning 3D Shape Completion under Weak Supervision.** International Journal of Computer Vision (2018).

Note that the source code and/or data is based on the following projects for which separate licenses apply:

* [pyrender](https://github.com/griegler/pyrender)
* [pyfusion](https://github.com/griegler/pyfusion)
* [PyMCubes](https://github.com/pmneila/PyMCubes)

Copyright (c) 2018 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.
