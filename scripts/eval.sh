CHECKPOINT_FOLDER=/yourfolder
FUSION_STRATEGY=intermediate
DATASET_MODE=v2v
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --dataset_mode ${DATASET_MODE}