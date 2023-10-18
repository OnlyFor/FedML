SOURCE=.
ENTRY=train_entry.py
ENTRY_ARGS='-m $FEDML_MODEL_NAME -mc $FEDML_MODEL_CACHE_PATH -mi $FEDML_MODEL_INPUT_DIM -mo $MODEL_OUTPUT_DIM -dn $FEDML_DATASET_NAME -dt $FEDML_DATASET_TYPE -dp $FEDML_DATASET_PATH'
CONFIG=config
DEST=./mlops
MODEL_NAME=lr
MODEL_CACHE=~/fedml_models
MODEL_INPUT_DIM=784
MODEL_OUTPUT_DIM=10
DATASET_NAME=mnist
DATASET_TYPE=csv
DATASET_PATH=~/fedml_data
fedml train build -sf $SOURCE -ep $ENTRY -ea "$ENTRY_ARGS" \
      -cf $CONFIG -df $DEST -m $MODEL_NAME -mc $MODEL_CACHE \
      -mi $MODEL_INPUT_DIM -mo $MODEL_OUTPUT_DIM -dn $DATASET_NAME -dt $DATASET_TYPE -dp $DATASET_PATH