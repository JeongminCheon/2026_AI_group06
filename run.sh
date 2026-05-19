# # step4-6
# python step5_1_make_threshold_quickchecks.py \
#   --prob-csv step3_6_seed_ensemble_mlp_outputs/test_public_probabilities.csv \
#   --out-dir step5_1_threshold_sweep_outputs_wide \
#   --prefix mlp3seed \
#   --start 0.40 \
#   --end 0.65 \
#   --step 0.02

# # step4-7
# # 이거 결과 확인 뒤, step4-6 실험을 다시 실행
# # 예시
# # python step5_1_make_threshold_quickchecks.py \
# #   --prob-csv step3_6_seed_ensemble_7seeds_outputs/test_public_probabilities.csv \
# #   --out-dir step5_1_threshold_sweep_7seeds_outputs \
# #   --prefix mlp7seed \
# #   --thresholds 0.45 0.48 0.51 0.54 0.57
# python step3_3_advanced_mlp.py \
#   --train-features step2_outputs/train_features.csv \
#   --test-features step2_outputs/test_public_features.csv \
#   --out-dir step5_0_seed_ensemble_7seeds_outputs \
#   --hidden-dims 512 256 128 \
#   --dropout 0.25 \
#   --lr 0.0007 \
#   --weight-decay 0.0001 \
#   --seeds 42 7 2025 123 777 3407 1004

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

