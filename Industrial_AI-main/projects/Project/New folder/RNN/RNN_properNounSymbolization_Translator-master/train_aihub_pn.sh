SRC_LANG='kr'
TRG_LANG='en'
BPE=1

RNN='mylstm'

DIM_WEMB=300
DIM_ENC=500
DIM_ATT=500 # better than 1000
DIM_DEC=500

TRAIN_FILE1='aihub_train.'$SRC_LANG'.shuf.tok.pn'
TRAIN_FILE2='aihub_train.'$TRG_LANG'.shuf.tok.pn'

DICT1='vocab.'$SRC_LANG'.pkl'
DICT2='vocab.'$TRG_LANG'.pkl'

#when use fi
DATA_DIR='./aihub/PN_version/'
VALID_FILE1='aihub_valid.'$SRC_LANG'.shuf.tok.pn'
VALID_FILE2='aihub_valid.'$TRG_LANG'.shuf.tok.pn'
SUB_DIR='subword'



VALID_FILE2=$DATA_DIR'/'$VALID_FILE2

#when BPE = 1
DATA_DIR=$DATA_DIR'/'$SUB_DIR
TRAIN_FILE1=$TRAIN_FILE1'.sub'
TRAIN_FILE2=$TRAIN_FILE2'.sub'
VALID_FILE1=$VALID_FILE1'.sub'

TRAIN_FILE1=$DATA_DIR'/'$TRAIN_FILE1
TRAIN_FILE2=$DATA_DIR'/'$TRAIN_FILE2
VALID_FILE1=$DATA_DIR'/'$VALID_FILE1

DICT1=$DATA_DIR'/'$DICT1
DICT2=$DATA_DIR'/'$DICT2

SAVE_DIR='./results'
#mkdir $SAVE_DIR

LOADMODEL=0

MODEL_FILE=$SRC_LANG'2'$TRG_LANG'.'$RNN'.'$DIM_WEMB'.'$DIM_ENC'.'$DIM_ATT'.'$DIM_DEC'.aihub'.'pn'

CUDA_VISIBLE_DEVICES=$1 python3 nmt_run.py --train=1 --rnn_name=$RNN \
        --save_dir=$SAVE_DIR --model_file=$MODEL_FILE \
        --train_src_file=$TRAIN_FILE1 --train_trg_file=$TRAIN_FILE2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 \
        --dim_wemb=$DIM_WEMB --dim_enc=$DIM_ENC --dim_att=$DIM_ATT --dim_dec=$DIM_DEC \
        --src_dict=$DICT1 --trg_dict=$DICT2 --load_model=$LOADMODEL #--print_every=10 --valid_every=100
