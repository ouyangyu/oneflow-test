## 测试环境
所有的测试都是在配置8张TITAN V-12066MiB GPU的服务器中进行，主要硬软件配置如下：
- `nvidia-smi topo -m`  
```
	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	CPU Affinity
GPU0	 X 	PIX	PHB	PHB	SYS	SYS	SYS	SYS	0-13,28-41
GPU1	PIX	 X 	PHB	PHB	SYS	SYS	SYS	SYS	0-13,28-41
GPU2	PHB	PHB	 X 	PIX	SYS	SYS	SYS	SYS	0-13,28-41
GPU3	PHB	PHB	PIX	 X 	SYS	SYS	SYS	SYS	0-13,28-41
GPU4	SYS	SYS	SYS	SYS	 X 	PIX	PHB	PHB	14-27,42-55
GPU5	SYS	SYS	SYS	SYS	PIX	 X 	PHB	PHB	14-27,42-55
GPU6	SYS	SYS	SYS	SYS	PHB	PHB	 X 	PIX	14-27,42-55
GPU7	SYS	SYS	SYS	SYS	PHB	PHB	PIX	 X 	14-27,42-55

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

- ### PipeDream
  - 分支: [pipedream@cad624f7](https://github.com/msr-fiddle/pipedream)
  - 实验步骤：
    - cp并修改`runtime/image_classification/driver_configs/alexnet_8dp.yml`
    - cp并修改`runtime/image_classification/driver_configs/alexnet_8pipedream.yml`
    - cp并修改`runtime/image_classification/driver_configs/vgg16_8pipedream.yml`
    - cp并修改`runtime/image_classification/driver_configs/vgg16_8dp.yml`
  - #### 实验汇总
    所有实验都是在单机8卡情况下进行，分2个stage，rank比例为：7:1
    |    类型             | GPU Memory |  单个Epoch时间     |  日志 |
    | ------------------ | -----------|-------------------| ------------ |
    | alexnet_8dp        |  4679MiB   | Epoch 0: 151.468 seconds、Epoch 1: 144.396 seconds |[pipedream_alexnet_8dp_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/pipedream_alexnet_8dp_logs.zip)  |
    | alexnet_8pipedream |  5047MiB   | Epoch 0: 105.446 seconds、Epoch 1: 102.291 seconds |[pipedream_alexnet_8pipedream_logs](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/pipedream_alexnet_8pipedream_logs.zip)|
    | vgg16_8dp          | 11845MiB    |  Epoch 0: 2443.655 seconds、Epoch 1: 2430.059 seconds |[pipedream_vgg16_dp_8_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/pipedream_vgg16_dp_8_logs.zip)   |
    | vgg16_8pipedream   |     |         |
  - #### vgg16_8dp
    - 在`pipedream/runtime`目录下，运行`python driver.py --config_file image_classification/driver_configs/vgg16_8dp.yml --launch_single_container --mount_directories /DATA/disk1/ImageNet/extract /home/oyy`
    - 加载真实数据
    - `watch nvidia-smi`信息如下：
        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     35961      C   /opt/conda/bin/python                      11769MiB |
        |    1     35962      C   /opt/conda/bin/python                      11845MiB |
        |    2     35963      C   /opt/conda/bin/python                      11845MiB |
        |    3     35964      C   /opt/conda/bin/python                      11845MiB |
        |    4     35965      C   /opt/conda/bin/python                      11833MiB |
        |    5     35966      C   /opt/conda/bin/python                      11849MiB |
        |    6     35967      C   /opt/conda/bin/python                      11849MiB |
        |    7     35968      C   /opt/conda/bin/python                      11849MiB |
        +-----------------------------------------------------------------------------+
        ```
  - #### alexnet_8dp
    - 在`pipedream/runtime`目录下，运行`python driver.py --config_file image_classification/driver_configs/alexnet_8dp.yml --launch_single_container --mount_directories /DATA/disk1/ImageNet/extract /home/oyy`
    - 合成数据
    - `watch nvidia-smi`信息如下：
        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     47348      C   /opt/conda/bin/python            4667MiB |
        |    1   N/A  N/A     47349      C   /opt/conda/bin/python            4679MiB |
        |    2   N/A  N/A     47350	 C   /opt/conda/bin/python            4679MiB |
        |    3   N/A  N/A     47351	 C   /opt/conda/bin/python            4679MiB |
        |    4   N/A  N/A     47352	 C   /opt/conda/bin/python            4667MiB |
        |    5   N/A  N/A     47353      C   /opt/conda/bin/python            4679MiB |
        |    6   N/A  N/A     47354	 C   /opt/conda/bin/python            4679MiB |
        |    7   N/A  N/A     47355	 C   /opt/conda/bin/python            4679MiB |
        +-----------------------------------------------------------------------------+
        ```
  - #### alexnet_8pipedream
    - 在`pipedream/runtime`目录下，运行`python driver.py --config_file image_classification/driver_configs/alexnet_8pipedream.yml --launch_single_container --mount_directories /DATA/disk1/ImageNet/extract /home/oyy`
    - 合成数据
    - `watch nvidia-smi`信息如下：
        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     56257	 C   /opt/conda/bin/python            5047MiB |
        |    0   N/A  N/A     56259	 C   /opt/conda/bin/python             683MiB |
        |    0   N/A  N/A     56262	 C   /opt/conda/bin/python             683MiB |
        |    1   N/A  N/A     56258	 C   /opt/conda/bin/python            4871MiB |
        |    2   N/A  N/A     56259	 C   /opt/conda/bin/python            4871MiB |
        |    3   N/A  N/A     56260	 C   /opt/conda/bin/python            4871MiB |
        |    4   N/A  N/A     56261	 C   /opt/conda/bin/python            4871MiB |
        |    5   N/A  N/A     56262	 C   /opt/conda/bin/python            5047MiB |
        |    6   N/A  N/A     56263	 C   /opt/conda/bin/python            5047MiB |
        |    7   N/A  N/A     56264	 C   /opt/conda/bin/python            2151MiB |
        +-----------------------------------------------------------------------------+
        ```

- ### OneFlow
  - 分支: [dev_ssp@dbab058b9](https://github.com/Oneflow-Inc/oneflow/tree/dev_ssp)
  - OneFlow-Benchmark: [test_ssp_tmp_branch@c66d85765](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/test_ssp_tmp_branch)

- #### ResNet50
  - ##### 实验汇总
    所有实验都是在2机8卡情况下进行，共16卡
    |    类型             | 数据集|             GPU                     |  单个Epoch时间     |  GPU Memory |
    | ------------------ | ---------|---------------------------|-------------------| ------------ |
    | ResNet50_dp        | 真实数据  |16卡                                 | 281.537 seconds  |    7877MiB   |
    | ResNet50_ssp_1     | 真实数据  |1个stage                             | 287.0098 seconds |     7877MiB   |
    | ResNet50_ssp_2     | 真实数据  |2个stage,0_w:1_w=1:30,placement相同   |         |8325MiB |
    | ResNet50_ssp_2_10  | 真实数据  |2个stage,0_w:1_w=1:10,placement相同   |         | |

  - ##### ResNet50_ssp_2：2机8卡SSP实验，2个stage,权重比为1:30
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 0:0-7,1:0-7.0:0-7,1:0-7 resnet50 /DATA/disk1/ImageNet/ofrecord/train 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0      7872      C   python3                                     8085MiB |
        |    1      7872      C   python3                                     8315MiB |
        |    2      7872      C   python3                                     8317MiB |
        |    3      7872      C   python3                                     8313MiB |
        |    4      7872      C   python3                                     8311MiB |
        |    5      7872      C   python3                                     8319MiB |
        |    6      7872      C   python3                                     8323MiB |
        |    7      7872      C   python3                                     8325MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     33341	 C   python3                          8097MiB |
        |    1   N/A  N/A     33341	 C   python3                          8325MiB |
        |    2   N/A  N/A     33341	 C   python3                          8309MiB |
        |    3   N/A  N/A     33341	 C   python3                          8309MiB |
        |    4   N/A  N/A     33341	 C   python3                          8323MiB |
        |    5   N/A  N/A     33341	 C   python3                          8315MiB |
        |    6   N/A  N/A     33341	 C   python3                          8325MiB |
        |    7   N/A  N/A     33341	 C   python3                          8319MiB |
        +-----------------------------------------------------------------------------+

        ```

  - ##### ResNet50_ssp_1：2机8卡SSP实验，1个stage
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 0:0-7,1:0-7.0:0-7,1:0-7 resnet50 /DATA/disk1/ImageNet/ofrecord/train 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     54559      C   python3                                     7651MiB |
        |    1     54559      C   python3                                     7869MiB |
        |    2     54559      C   python3                                     7873MiB |
        |    3     54559      C   python3                                     7877MiB |
        |    4     54559      C   python3                                     7861MiB |
        |    5     54559      C   python3                                     7869MiB |
        |    6     54559      C   python3                                     7861MiB |
        |    7     54559      C   python3                                     7861MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     23494	 C   python3                          7635MiB |
        |    1   N/A  N/A     23494	 C   python3                          7861MiB |
        |    2   N/A  N/A     23494	 C   python3                          7867MiB |
        |    3   N/A  N/A     23494	 C   python3                          7869MiB |
        |    4   N/A  N/A     23494	 C   python3                          7877MiB |
        |    5   N/A  N/A     23494	 C   python3                          7861MiB |
        |    6   N/A  N/A     23494	 C   python3                          7867MiB |
        |    7   N/A  N/A     23494	 C   python3                          7869MiB |
        +-----------------------------------------------------------------------------+
        ```
  - ##### ResNet50_dp：2机8卡dp实验
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 '' resnet50 /DATA/disk1/ImageNet/ofrecord/train 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0      1915      C   python3                                     7635MiB |
        |    1      1915      C   python3                                     7861MiB |
        |    2      1915      C   python3                                     7867MiB |
        |    3      1915      C   python3                                     7861MiB |
        |    4      1915      C   python3                                     7869MiB |
        |    5      1915      C   python3                                     7867MiB |
        |    6      1915      C   python3                                     7861MiB |
        |    7      1915      C   python3                                     7877MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     27618	 C   python3                          7635MiB |
        |    1   N/A  N/A     27618	 C   python3                          7861MiB |
        |    2   N/A  N/A     27618	 C   python3                          7877MiB |
        |    3   N/A  N/A     27618	 C   python3                          7873MiB |
        |    4   N/A  N/A     27618	 C   python3                          7861MiB |
        |    5   N/A  N/A     27618	 C   python3                          7861MiB |
        |    6   N/A  N/A     27618	 C   python3                          7861MiB |
        |    7   N/A  N/A     27618	 C   python3                          7861MiB |
        +-----------------------------------------------------------------------------+
        ```
