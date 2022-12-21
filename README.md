# Usage

## Main Dependencies

```bash
python>=3.5
pytorch>=1.0.0
opencv
numpy
```

The hardware environment is NVIDIA GTX 1080Ti.

## Prepare Dataset

We need to prepare 'file\_list.txt', the data format inside is as follows:

```bash
/data/oulu/spatial/Train_npynew/1_1_01_1.npy /data/oulu/depth/Train_rppg/1_1_01_1.npy 1
```

the `/data/oulu/spatial/Train_npynew/1_1_01_1.npy` is the location of the processed video, and `//data/oulu/depth/Train_rppg/1_1_01_1.npy` is the position corresponding to the rppg signal. `1` is label.

## Train and Test

### Train

You can modify the configuration according to your needs by modifying `options.py` . And run :

```bash
python main.py
```

to start a train.

The directory of the trained model is determined by `--checkpoints_dir` and `--name` in `options.py`.

The `--model` in `options.py` represents three different models, you can get specific details from the ablation experiment of the paper.

If there is a `CUDA out of memory` problem, you can modify `gpu_ids` or `batch_size` in `options.py`.

### Test

We provide three models in the `checkpoints`. You can run:

```bash
python test.py  --name model1
```

```bash
python test.py  --name model2
```

```bash
python test.py  --name model2
```

to test different models.

# Pre Process

```bash
python video2npy.py --dir=1 --threads=10 --width=32 --height=32

python gen_protocols.py --protocol=Protocol_4
```

# Experiment

| Date  | Name      | Protocol | GPU | batch size | Epoch | Status  | Note                                                                                |
| ----- | --------- | -------- | --- | :--------- | ----- | ------- | ----------------------------------------------------------------------------------- |
| 12/19 | 1219-1    | Oulu-P1  | 1   | 5          | 30    | Succ    | 成功跑完                                                                                |
| 12/19 | 1219-p2-1 | Oulu-P2  | 1   | 5          | 30    | Succ    | 成功跑完                                                                                |
| 12/19 | 1219-p2-5 | Oulu-P2  | 4   | 24         | 30    | Succ    | 成功跑完                                                                                |
| 12/19 | 1219-p2-6 | Oulu-P2  | 4   | 20         | 60    | Fail    | 跑到第16个epoch，主线程直接gpu-util 100%，程序卡死                                                 |
| 12/19 | 1219-p2-7 | Oulu-P2  | 4   | 24         | 60    | Fail    |                                                                                     |
| 12/19 | 1219-p1-2 | Oulu-P1  | 1   | 5          | 60    | Succ    |                                                                                     |
| 12/19 | 1219-p2-8 | Oulu-P2  | 1   | 5          | 60    | Succ    |                                                                                     |
| 12/19 | 1219-p2-9 | Oulu-P2  | 2   | 12         | 60    | Fail    | 并行化修改后；第27个EPOCH时RuntimeError: CUDA error: an illegal memory access was encountered |
| 12/20 | 1220-p2-1 | Oulu-P2  | 2   | 12         | 60    | Running | 并行化修改后；                                                                             |
|       |           |          |     |            |       |         |                                                                                     |

# PS

1.  多卡训练时，batch size设置的太小，容易出现GPU-Util 100%程序卡死的情况。

2.  batch size设置太大，会出现:

```bash
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

