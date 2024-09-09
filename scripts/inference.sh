# After launching the docker container 
EVAL_DIR='checkpoints/basemodel/checkpoints/'
CHECKPOINT='checkpoints/basemodel/checkpoints/model.bin'
VIDEO='/root/raw_mmau/train3/014462.mp4'
CUDA_VISIBLE_DEVICES=4 python -m pdb src/tasks/run_adapt_inference.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --test_video_fname $VIDEO \
       --do_lower_case \
       --do_test 
