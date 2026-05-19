# step0
python step0_data_check.py \
  --data-root ./26S_AI536_NE450 \
  --out-dir step0_outputs

# step1
python step1_eda.py \
  --step0-dir step0_outputs \
  --out-dir step1_outputs

# step2
python step2_extract_features.py \
  --data-root ./26S_AI536_NE450 \
  --out-dir step2_outputs

# step3-1
python step3_1_train_tabular_baselines.py \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step3_1_outputs \
  --n-splits 5

# step3-2
python step3_2_train_mlp.py \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step3_2_mlp_outputs

# step3-3
python step3_3_advanced_mlp.py \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step3_3_wide_mlp_512_256_128_64_outputs \
  --hidden-dims 512 256 128 64 \
  --dropout 0.25 \
  --lr 0.0007 \
  --weight-decay 0.0001 \
  --seeds 42

# step3-4
python step3_3_advanced_mlp.py \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step3_4_multitask_mlp_outputs \
  --hidden-dims 512 256 128 \
  --dropout 0.25 \
  --aux-regression \
  --reg-alpha 0.1 \
  --lr 0.0007 \
  --weight-decay 0.0001 \
  --seeds 42

# step3-5
python step3_3_advanced_mlp.py \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step3_5_corr_ffr_top300_mlp_outputs \
  --hidden-dims 512 256 128 \
  --dropout 0.25 \
  --feature-selection corr_ffr \
  --top-k 300 \
  --lr 0.0007 \
  --weight-decay 0.0001 \
  --seeds 42

# step3-6
python step3_3_advanced_mlp.py \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step3_6_seed_ensemble_mlp_outputs \
  --hidden-dims 512 256 128 \
  --dropout 0.25 \
  --lr 0.0007 \
  --weight-decay 0.0001 \
  --seeds 42 7 2025

# step4-1
python step4_1_train_cnn_models.py \
  --model unet_encoder \
  --data-root ./26S_AI536_NE450 \
  --out-dir step4_6a_unet_encoder_s128_outputs \
  --num-slices 128 \
  --image-size 128 \
  --base-ch 24 \
  --embed-dim 256 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda

# step4-2
python step4_1_train_cnn_models.py \
  --model simple_cnn \
  --data-root ./26S_AI536_NE450 \
  --out-dir step4_6b_simple_cnn_large_s128_outputs \
  --num-slices 128 \
  --image-size 128 \
  --base-ch 24 \
  --embed-dim 256 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda

# step4-3
python step4_1_train_cnn_models.py \
  --model cnn3d \
  --data-root ./26S_AI536_NE450 \
  --out-dir step4_6c_3dcnn_large_outputs \
  --depth-size 64 \
  --image-size 128 \
  --base-ch 12 \
  --embed-dim 256 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda

# step4-4
python step4_2_train_transformer_models.py \
  --model slice_vit \
  --vit-name deit_tiny_patch16_224 \
  --data-root ./26S_AI536_NE450 \
  --out-dir step4_7a_deit_tiny_s32_outputs \
  --num-slices 32 \
  --image-size 224 \
  --embed-dim 128 \
  --pooling mean_max \
  --batch-size 1 \
  --eval-batch-size 1 \
  --epochs 80 \
  --patience 15 \
  --augment \
  --device cuda

# step4-5
python step4_2_train_transformer_models.py \
  --model seq_transformer \
  --data-root ./26S_AI536_NE450 \
  --out-dir step4_7b_seq_transformer_seq512_dim128_outputs \
  --seq-len 512 \
  --embed-dim 128 \
  --num-heads 4 \
  --num-layers 4 \
  --ff-dim 256 \
  --seq-pooling mean_max \
  --batch-size 8 \
  --eval-batch-size 16 \
  --epochs 120 \
  --patience 25 \
  --augment \
  --device cuda

# ---------------------------------------------------
# 피드백 후 새롭게 다시
# ---------------------------------------------------

# step4-6
python step5_1_make_threshold_quickchecks.py \
  --prob-csv step3_6_seed_ensemble_mlp_outputs/test_public_probabilities.csv \
  --out-dir step5_1_threshold_sweep_outputs_wide \
  --prefix mlp3seed \
  --start 0.40 \
  --end 0.65 \
  --step 0.02

# step4-7
# 이거 결과 확인 뒤, step4-6 실험을 다시 실행
# 예시
# python step5_1_make_threshold_quickchecks.py \
#   --prob-csv step3_6_seed_ensemble_7seeds_outputs/test_public_probabilities.csv \
#   --out-dir step5_1_threshold_sweep_7seeds_outputs \
#   --prefix mlp7seed \
#   --thresholds 0.45 0.48 0.51 0.54 0.57
python step3_3_advanced_mlp.py \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step5_0_seed_ensemble_7seeds_outputs \
  --hidden-dims 512 256 128 \
  --dropout 0.25 \
  --lr 0.0007 \
  --weight-decay 0.0001 \
  --seeds 42 7 2025 123 777 3407 1004

# step4-8
python step5_2_probability_ensemble.py \
  --inputs \
    step5_0_seed_ensemble_7seeds_outputs/test_public_probabilities.csv \
    step3_1_outputs/extra_trees_test_public_probabilities.csv \
    step3_1_outputs/random_forest_test_public_probabilities.csv \
  --weights 0.8 0.15 0.05 \
  --out-dir step5_2_feature_ensemble_outputs \
  --prefix mlp080_et015_rf005 \
  --thresholds 0.45 0.48 0.51 0.54 0.57

# step4-9
python step5_3_train_attention_mil.py \
  --data-root ./26S_AI536_NE450 \
  --out-dir step5_3_attention_s32_outputs \
  --num-slices 32 \
  --image-size 128 \
  --base-ch 16 \
  --embed-dim 128 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda

python step5_3_train_attention_mil.py \
  --data-root ./26S_AI536_NE450 \
  --out-dir step5_3_attention_s64_outputs \
  --num-slices 64 \
  --image-size 128 \
  --base-ch 16 \
  --embed-dim 128 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda

python step5_3_train_attention_mil.py \
  --data-root ./26S_AI536_NE450 \
  --out-dir step5_3_attention_s64_large_outputs \
  --num-slices 64 \
  --image-size 128 \
  --base-ch 24 \
  --embed-dim 256 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda

# step5
python step6_1_make_soft_ensemble_quickchecks.py \
  --prob-3seed step3_6_seed_ensemble_mlp_outputs/test_public_probabilities.csv \
  --prob-7seed step5_0_seed_ensemble_7seeds_outputs/test_public_probabilities.csv \
  --prob-attn step5_3_attention_s64_outputs/test_public_probabilities.csv \
  --out-dir step6_1_soft_ensemble_outputs

# step6
python step6_2_make_a5_finetune_quickchecks.py \
  --prob-3seed step3_6_seed_ensemble_mlp_outputs/test_public_probabilities.csv \
  --prob-7seed step5_0_seed_ensemble_7seeds_outputs/test_public_probabilities.csv \
  --out-dir step6_2_a5_finetune_outputs

# step7


# step8


# step9


# step10


