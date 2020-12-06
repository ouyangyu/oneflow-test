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
    - alexnet batch size为256
    - vgg16的batch size为32(vgg16_8pipedream 时batch size为64会oom)
    |    类型             | GPU Memory |  单个Epoch时间     |  日志 |
    | ------------------ | -----------|-------------------| ------------ |
    | alexnet_8dp        |  4679MiB   | Epoch 0: 151.468 seconds、Epoch 1: 144.396 seconds |[pipedream_alexnet_8dp_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/pipedream_alexnet_8dp_logs.zip)  |
    | alexnet_8pipedream |  5047MiB   | Epoch 0: 105.446 seconds、Epoch 1: 102.291 seconds |[pipedream_alexnet_8pipedream_logs](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/pipedream_alexnet_8pipedream_logs.zip)|
    | vgg16_8dp_b64          | 11845MiB    |  Epoch 0: 2443.655 seconds、Epoch 1: 2430.059 seconds |[pipedream_vgg16_dp_8_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/pipedream_vgg16_dp_8_logs.zip)   |
    | vgg16_8pipedream_b32   |  7907MiB   |   Epoch 0: 1644.832 seconds   | [vgg16_8pipedream_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/vgg16_8pipedream_logs.zip)|
    | vgg16_8dp_b32          | 11687MiB    |Epoch 0: 2647.071 seconds  |[pipedream_vgg_8dp_b32_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/pipedream/js/pipedream_vgg_8dp_b32_logs.zip)   |
  
  - #### vgg16_8pipedream
    - 在`pipedream/runtime`目录下，运行`python driver.py --config_file image_classification/driver_configs/vgg16_8pipedream.yml --launch_single_container --mount_directories /DATA/disk1/ImageNet/extract /home/oyy`
    - 加载真实数据
    - `watch nvidia-smi`信息如下：
        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     39304      C   /opt/conda/bin/python                       7907MiB |
        |    0     39306      C   /opt/conda/bin/python                        681MiB |
        |    0     39307      C   /opt/conda/bin/python                        681MiB |
        |    0     39308      C   /opt/conda/bin/python                        681MiB |
        |    0     39309      C   /opt/conda/bin/python                        681MiB |
        |    0     39310      C   /opt/conda/bin/python                        681MiB |
        |    0     39311      C   /opt/conda/bin/python                        681MiB |
        |    1     39305      C   /opt/conda/bin/python                       7907MiB |
        |    2     39306      C   /opt/conda/bin/python                       7887MiB |
        |    3     39307      C   /opt/conda/bin/python                       7887MiB |
        |    4     39308      C   /opt/conda/bin/python                       7907MiB |
        |    5     39309      C   /opt/conda/bin/python                       7907MiB |
        |    6     39310      C   /opt/conda/bin/python                       7887MiB |
        |    7     39311      C   /opt/conda/bin/python                       5439MiB |
        +-----------------------------------------------------------------------------+
        ```

- #### vgg16_8dp batch size为32
    - 在`pipedream/runtime`目录下，运行`python driver.py --config_file image_classification/driver_configs/vgg16_8dp.yml --launch_single_container --mount_directories /DATA/disk1/ImageNet/extract /home/oyy`
    - 加载真实数据
    - `watch nvidia-smi`信息如下：
        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     46456      C   /opt/conda/bin/python                      11675MiB |
        |    1     46457      C   /opt/conda/bin/python                      11687MiB |
        |    2     46458      C   /opt/conda/bin/python                      11687MiB |
        |    3     46459      C   /opt/conda/bin/python                      11687MiB |
        |    4     46460      C   /opt/conda/bin/python                      11675MiB |
        |    5     46461      C   /opt/conda/bin/python                      11687MiB |
        |    6     46462      C   /opt/conda/bin/python                      11687MiB |
        |    7     46463      C   /opt/conda/bin/python                      11687MiB |
        +-----------------------------------------------------------------------------+
        ```

  - #### vgg16_8dp batch size为64
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

- #### 数据汇总
  ssp情况下都是stage0位0-6号卡，stage1为7号卡
  AlexNet 的batch size 是为252，合成数据
  vgg16 的batch size 是32，真实数据
  | 模型类型     |  stage权重比           |   显存占用           | 单个epoch时间 |  日志地址  |
  | -----------| --------------------- | ------------------- | --------------- | ---------|
  |vgg16_ssp   |stage0_w:stage1_w=11:7 |9335MiB/4499MiB(07卡)| 1351.686 seconds| [of_vgg16_ssp_1_8_stage2_11-7_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/js/of_vgg16_ssp_1_8_stage2_11-7_logs.zip)|
  |alexnet_ssp |stage0_w:stage1_w=3:1  |4953MiB、969MiB(07卡) | 85.51 seconds |  [of_alexnet_ssp_1_8_stage2_3-1_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/js/of_alexnet_ssp_1_8_stage2_3-1_logs.zip) | 
  | vgg16_dp   | --                    | 8639MiB            |  1847.705 seconds  |[of_vgg16_dp_b28_1_8_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/js/of_vgg16_dp_b28_1_8_logs.zip) | 
  | alexnet_dp | --                    | 3535MiB            | 76.251 seconds | [of_alexnet_dp_1_8_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/js/of_alexnet_dp_1_8_logs.zip) | 

- #### vgg16_ssp_b28
  - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 28 0:0-6.0:7 vgg /DATA/disk1/ImageNet/ofrecord/train`
  - `watch nvidia-smi`信息如下：
    ```
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.64	  Driver Version: 440.64       CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN V             Off  | 00000000:04:00.0 Off |                  N/A |
    | 35%   54C    P2   102W / 250W |   9247MiB / 12066MiB |     76%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:05:00.0 Off |                  N/A |
    | 41%   61C    P2    59W / 250W |   9353MiB / 12066MiB |     48%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  TITAN V             Off  | 00000000:08:00.0 Off |                  N/A |
    | 39%   58C    P2    48W / 250W |   9353MiB / 12066MiB |     89%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  TITAN V             Off  | 00000000:09:00.0 Off |                  N/A |
    | 44%   63C    P2    49W / 250W |   9353MiB / 12066MiB |     51%      Default |
    +-------------------------------+----------------------+----------------------+
    |   4  TITAN V             Off  | 00000000:85:00.0 Off |                  N/A |
    | 40%   59C    P2   103W / 250W |   9353MiB / 12066MiB |     89%      Default |
    +-------------------------------+----------------------+----------------------+
    |   5  TITAN V             Off  | 00000000:86:00.0 Off |                  N/A |
    | 43%   63C    P2   160W / 250W |   9353MiB / 12066MiB |     47%      Default |
    +-------------------------------+----------------------+----------------------+
    |   6  TITAN V             Off  | 00000000:89:00.0 Off |                  N/A |
    | 35%   54C    P2   167W / 250W |   9329MiB / 12066MiB |     46%      Default |
    +-------------------------------+----------------------+----------------------+
    |   7  TITAN V             Off  | 00000000:8A:00.0 Off |                  N/A |
    | 32%   49C    P2    42W / 250W |   4517MiB / 12066MiB |     13%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      9335      C   python3                                     9229MiB |
    |    1      9335      C   python3                                     9335MiB |
    |    2      9335      C   python3                                     9335MiB |
    |    3      9335      C   python3                                     9335MiB |
    |    4      9335      C   python3                                     9335MiB |
    |    5      9335      C   python3                                     9335MiB |
    |    6      9335      C   python3                                     9311MiB |
    |    7      9335      C   python3                                     4499MiB |
    +-----------------------------------------------------------------------------+
    ```

- #### vgg16_dp_b28
  - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 28 '' vgg /DATA/disk1/ImageNet/ofrecord/train`
  - `watch nvidia-smi`信息如下：
    ```
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     29500	 C   python3                          6349MiB |
    |    1   N/A  N/A     29500	 C   python3                          8293MiB |
    |    2   N/A  N/A     29500	 C   python3                          8293MiB |
    |    3   N/A  N/A     29500	 C   python3                          8293MiB |
    |    4   N/A  N/A     29500	 C   python3                          8293MiB |
    |    5   N/A  N/A     29500	 C   python3                          8293MiB |
    |    6   N/A  N/A     29500	 C   python3                          8293MiB |
    |    7   N/A  N/A     29500	 C   python3                          8293MiB |
    +-----------------------------------------------------------------------------+
    ```

- #### AlexNet ssp
  - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 252 0:0-6.0:7 alexnet`
  - `watch nvidia-smi`信息如下：
    ```
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.64	  Driver Version: 440.64       CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN V             Off  | 00000000:04:00.0 Off |                  N/A |
    | 30%   47C    P2    90W / 250W |   4633MiB / 12066MiB |     86%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:05:00.0 Off |                  N/A |
    | 35%   54C    P2    44W / 250W |   4965MiB / 12066MiB |     88%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  TITAN V             Off  | 00000000:08:00.0 Off |                  N/A |
    | 33%   51C    P2   111W / 250W |   4965MiB / 12066MiB |     88%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  TITAN V             Off  | 00000000:09:00.0 Off |                  N/A |
    | 36%   55C    P2   121W / 250W |   4967MiB / 12066MiB |     87%      Default |
    +-------------------------------+----------------------+----------------------+
    |   4  TITAN V             Off  | 00000000:85:00.0 Off |                  N/A |
    | 34%   52C    P2   118W / 250W |   4965MiB / 12066MiB |     88%      Default |
    +-------------------------------+----------------------+----------------------+
    |   5  TITAN V             Off  | 00000000:86:00.0 Off |                  N/A |
    | 36%   55C    P2    56W / 250W |   4965MiB / 12066MiB |     89%      Default |
    +-------------------------------+----------------------+----------------------+
    |   6  TITAN V             Off  | 00000000:89:00.0 Off |                  N/A |
    | 29%   46C    P2   112W / 250W |   4941MiB / 12066MiB |     76%      Default |
    +-------------------------------+----------------------+----------------------+
    |   7  TITAN V             Off  | 00000000:8A:00.0 Off |                  N/A |
    | 28%   43C    P2    58W / 250W |    981MiB / 12066MiB |      9%      Default |
    +-------------------------------+----------------------+----------------------+
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      8897	 C   python3                          4621MiB |
    |    1   N/A  N/A      8897	 C   python3                          4953MiB |
    |    2   N/A  N/A      8897	 C   python3                          4953MiB |
    |    3   N/A  N/A      8897	 C   python3                          4953MiB |
    |    4   N/A  N/A      8897	 C   python3                          4953MiB |
    |    5   N/A  N/A      8897	 C   python3                          4953MiB |
    |    6   N/A  N/A      8897	 C   python3                          4929MiB |
    |    7   N/A  N/A      8897	 C   python3                           969MiB |
    +-----------------------------------------------------------------------------+
    ```

- #### AlexNet dp
  - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 252 '' alexnet`
  - `watch nvidia-smi`信息如下：
    ```
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     11829	 C   python3                          2789MiB |
    |    1   N/A  N/A     11829      C   python3                          3535MiB |
    |    2   N/A  N/A     11829      C   python3                          3535MiB |
    |    3   N/A  N/A     11829	 C   python3                          3535MiB |
    |    4   N/A  N/A     11829      C   python3                          3535MiB |
    |    5   N/A  N/A     11829      C   python3                          3535MiB |
    |    6   N/A  N/A     11829	 C   python3                          3535MiB |
    |    7   N/A  N/A     11829      C   python3                          3535MiB |
    +-----------------------------------------------------------------------------+
    ```

- #### ResNet50@abb3c0a2d
  - #####  ResNet50_ssp2_1-5  2个stage 比例为1:5 合成数据
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 0:0-7,1:0-7.0:0-7,1:0-7 resnet50 '' 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     52736      C   python3                                     8747MiB |
        |    1     52736      C   python3                                     8975MiB |
        |    2     52736      C   python3                                     8975MiB |
        |    3     52736      C   python3                                     8993MiB |
        |    4     52736      C   python3                                     8983MiB |
        |    5     52736      C   python3                                     8983MiB |
        |    6     52736      C   python3                                     8991MiB |
        |    7     52736      C   python3                                     8973MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     19398      C   python3                          8749MiB |
        |    1   N/A  N/A     19398      C   python3                          8983MiB |
        |    2   N/A  N/A     19398      C   python3                          8975MiB |
        |    3   N/A  N/A     19398      C   python3                          8973MiB |
        |    4   N/A  N/A     19398      C   python3                          8973MiB |
        |    5   N/A  N/A     19398      C   python3                          8983MiB |
        |    6   N/A  N/A     19398	 C   python3                          8987MiB |
        |    7   N/A  N/A     19398      C   python3                          8983MiB |
        +-----------------------------------------------------------------------------+
        ```
      - 结果：

- #### ResNet50@46dd15904
  - ##### 实验汇总
    所有实验都是在2机8卡情况下进行，共16卡
    |    类型             | 数据集|             GPU                     |  单个Epoch时间     |  GPU Memory |  日志 |
    | ------------------ | ---------|---------------------------|-------------------| ------------ | ------------|
    | ResNet50_dp        | 真实数据  |16卡                                 | 281.537 seconds  | 7877MiB   |[of_resnet50_dp_2_8_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/of_resnet50_dp_2_8_logs.zip) |
    | ResNet50_ssp1_1     | 真实数据  |1个stage                             | 287.0098 seconds | 7877MiB   |[oneflow_ssp/of_resnet50_ssp_stage-1_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/of_resnet50_ssp_stage-1_logs.zip)|
    | ResNet50_ssp2_1-30  | 真实数据  |2个stage,0_w:1_w=1:30,placement相同   | 376.589 seconds |  8325MiB |[of_resnet50_ssp_stage-2_1-30_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/of_resnet50_ssp_stage-2_1-30_logs.zip)|
    | ResNet50_ssp2_1-10  | 真实数据  |2个stage,0_w:1_w=1:10,placement相同   | 330.723 seconds | 8867MiB |[of_resnet50_ssp_stage-2_1-10_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/of_resnet50_ssp_stage-2_1-10_logs.zip)|
    | ResNet50_dp         | 合成数据  |16卡                                 |  288.564 seconds  |  6735MiB | -|
    | ResNet50_ssp1_1     | 合成数据  |1个stage                             |  281.435 seconds | 6749MiB | [of_resnet50_ssp_stage-1_sd_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/of_resnet50_ssp_stage-1_sd_logs.zip) |
    | ResNet50_ssp2_1-10  | 合成数据  |2个stage,0_w:1_w=1:10,placement相同   |  324.634 seconds |  7857MiB |- |
    | ResNet50_ssp2_1-5  | 合成数据  |2个stage,0_w:1_w=1:5,placement相同   |    328.287 seconds |  8983MiB | [of_resnet50_ssp_stage-2_1-5_logs.zip](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_ssp/of_resnet50_ssp_stage-2_1-5_sd_logs.zip) |
  
  - #####  ResNet50_ssp2_1-5  2个stage 比例为1:5 合成数据
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 0:0-7,1:0-7.0:0-7,1:0-7 resnet50 '' 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     30316      C   python3                                     8747MiB |
        |    1     30316      C   python3                                     8973MiB |
        |    2     30316      C   python3                                     8977MiB |
        |    3     30316      C   python3                                     8973MiB |
        |    4     30316      C   python3                                     8973MiB |
        |    5     30316      C   python3                                     8979MiB |
        |    6     30316      C   python3                                     8991MiB |
        |    7     30316      C   python3                                     8973MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     56495	 C   python3                          8753MiB |
        |    1   N/A  N/A     56495	 C   python3                          8975MiB |
        |    2   N/A  N/A     56495	 C   python3                          8975MiB |
        |    3   N/A  N/A     56495	 C   python3                          8983MiB |
        |    4   N/A  N/A     56495	 C   python3                          8973MiB |
        |    5   N/A  N/A     56495	 C   python3                          8975MiB |
        |    6   N/A  N/A     56495	 C   python3                          8973MiB |
        |    7   N/A  N/A     56495	 C   python3                          8973MiB |
        +-----------------------------------------------------------------------------+
        ```

  - #####   ResNet50_ssp1_1  1个stage  合成数据
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 0:0-7,1:0-7.0:0-7,1:0-7 resnet50 '' 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     27065      C   python3                                     6509MiB |
        |    1     27065      C   python3                                     6739MiB |
        |    2     27065      C   python3                                     6735MiB |
        |    3     27065      C   python3                                     6745MiB |
        |    4     27065      C   python3                                     6735MiB |
        |    5     27065      C   python3                                     6741MiB |
        |    6     27065      C   python3                                     6745MiB |
        |    7     27065      C   python3                                     6745MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     53384      C   python3                          6509MiB |
        |    1   N/A  N/A     53384      C   python3                          6737MiB |
        |    2   N/A  N/A     53384      C   python3                          6739MiB |
        |    3   N/A  N/A     53384      C   python3                          6749MiB |
        |    4   N/A  N/A     53384	 C   python3                          6737MiB |
        |    5   N/A  N/A     53384      C   python3                          6745MiB |
        |    6   N/A  N/A     53384      C   python3                          6749MiB |
        |    7   N/A  N/A     53384	 C   python3                          6751MiB |
        +-----------------------------------------------------------------------------+
        ```

  - #####  ResNet50_ssp2_1-10 2个stage，1:10 合成数据
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 0:0-7,1:0-7.0:0-7,1:0-7 resnet50 '' 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     22346      C   python3                                     7615MiB |
        |    1     22346      C   python3                                     7835MiB |
        |    2     22346      C   python3                                     7843MiB |
        |    3     22346      C   python3                                     7833MiB |
        |    4     22346      C   python3                                     7837MiB |
        |    5     22346      C   python3                                     7837MiB |
        |    6     22346      C   python3                                     7857MiB |
        |    7     22346      C   python3                                     7835MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     48842      C   python3                          7629MiB |
        |    1   N/A  N/A     48842      C   python3                          7835MiB |
        |    2   N/A  N/A     48842      C   python3                          7833MiB |
        |    3   N/A  N/A     48842      C   python3                          7833MiB |
        |    4   N/A  N/A     48842      C   python3                          7839MiB |
        |    5   N/A  N/A     48842	 C   python3                          7847MiB |
        |    6   N/A  N/A     48842      C   python3                          7837MiB |
        |    7   N/A  N/A     48842      C   python3                          7833MiB |
        +-----------------------------------------------------------------------------+
        ```

  - ##### ResNet50_dp 合成数据
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 '' resnet50 '' 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     17765      C   python3                                     6513MiB |
        |    1     17765      C   python3                                     6751MiB |
        |    2     17765      C   python3                                     6749MiB |
        |    3     17765      C   python3                                     6737MiB |
        |    4     17765      C   python3                                     6737MiB |
        |    5     17765      C   python3                                     6737MiB |
        |    6     17765      C   python3                                     6737MiB |
        |    7     17765      C   python3                                     6735MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     43850	 C   python3                          6519MiB |
        |    1   N/A  N/A     43850	 C   python3                          6745MiB |
        |    2   N/A  N/A     43850	 C   python3                          6737MiB |
        |    3   N/A  N/A     43850	 C   python3                          6745MiB |
        |    4   N/A  N/A     43850	 C   python3                          6735MiB |
        |    5   N/A  N/A     43850	 C   python3                          6749MiB |
        |    6   N/A  N/A     43850	 C   python3                          6749MiB |
        |    7   N/A  N/A     43850	 C   python3                          6735MiB |
        +-----------------------------------------------------------------------------+

        ```


  - ##### ResNet50_ssp2_1-10：2机8卡SSP实验，2个stage,权重比为1:10
    - 在金山01和02的`OneFlow-Benchmark/Classification/cnns`目录下分别运行命令：`./train_fp32_ssp.sh 4 64 0:0-7,1:0-7.0:0-7,1:0-7 resnet50 /DATA/disk1/ImageNet/ofrecord/train 10.0.22.16,10.0.22.3`
    - `watch nvidia-smi`信息如下：

        ```
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID   Type   Process name                             Usage      |
        |=============================================================================|
        |    0     13242      C   python3                                     8643MiB |
        |    1     13242      C   python3                                     8859MiB |
        |    2     13242      C   python3                                     8857MiB |
        |    3     13242      C   python3                                     8867MiB |
        |    4     13242      C   python3                                     8859MiB |
        |    5     13242      C   python3                                     8857MiB |
        |    6     13242      C   python3                                     8861MiB |
        |    7     13242      C   python3                                     8867MiB |
        +-----------------------------------------------------------------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A     38936	 C   python3                          8637MiB |
        |    1   N/A  N/A     38936	 C   python3                          8859MiB |
        |    2   N/A  N/A     38936	 C   python3                          8867MiB |
        |    3   N/A  N/A     38936	 C   python3                          8863MiB |
        |    4   N/A  N/A     38936	 C   python3                          8857MiB |
        |    5   N/A  N/A     38936	 C   python3                          8861MiB |
        |    6   N/A  N/A     38936	 C   python3                          8857MiB |
        |    7   N/A  N/A     38936	 C   python3                          8867MiB |
        +-----------------------------------------------------------------------------+
        ```
  - ##### ResNet50_ssp2_1-30：2机8卡SSP实验，2个stage,权重比为1:30
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
