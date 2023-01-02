NAME='testscript2'

CHECKPOINTS='./checkpoints'
BATCH_SIZE=8
EPOCH=100
GPU='0'

TRAIN_FILE_LIST='/mnt/hdd.user/datasets/FAS/Oulu-NPU/Protocols/Protocol_1/Train_32_proto.txt'
DEV_FILE_LIST='/mnt/hdd.user/datasets/FAS/Oulu-NPU/Protocols/Protocol_1/Dev_32_proto.txt'
TEST_FILE_LIST='/mnt/hdd.user/datasets/FAS/Oulu-NPU/Protocols/Protocol_1/Test_32_proto.txt'

W_CLS=1
W_NP=5
W_L1=5
W_GAN=1

LR_D=3e-5
LR_G=3e-5
LR_C=3e-5

SEED=1023
DEBUG=1

# 新建输出文件夹
SAVE_DIR="$CHECKPOINTS/$NAME"
mkdir $SAVE_DIR

#拷贝运行脚本
cp "./run.sh" "$SAVE_DIR/run_$(TZ=UTC-8 date +%Y-%m-%d-%H:%M:%S).sh"

# 输出基本信息
echo "========================================= INFO ========================================="
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
echo "name: $NAME, batch size: $BATCH_SIZE, epoch: $EPOCH, GPU: $GPU"
echo "w_cls: $W_CLS, w_NP: $W_NP, w_L1: $W_L1, w_gan: $W_GAN"
echo "lr_D: $LR_D, lr_G: $LR_G, lr_C: $LR_C"
echo "seed: $SEED, debug: $DEBUG"
echo $TRAIN_FILE_LIST
echo $DEV_FILE_LIST
echo $TEST_FILE_LIST
echo "========================================================================================"

# 训练网络
python main.py --checkpoints_dir=$CHECKPOINTS \
--name=$NAME \
--batch_size=$BATCH_SIZE \
--epoch=$EPOCH \
--gpu_ids=$GPU \
--train_file_list=$TRAIN_FILE_LIST \
--dev_file_list=$DEV_FILE_LIST \
--test_file_list=$TEST_FILE_LIST \
--w_cls=$W_CLS \
--w_NP=$W_NP \
--w_L1=$W_L1 \
--w_gan=$W_GAN \
--lr_D=$LR_D \
--lr_G=$LR_G \
--lr_C=$LR_C \
--seed=$SEED \
--debug=$DEBUG