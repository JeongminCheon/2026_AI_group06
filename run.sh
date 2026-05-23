# step8-1
python step8_train_lesion_models.py \
  --mode lesion_cnn \
  --data-root ./26S_AI536_NE450 \
  --out-dir step8_1_lesion_cnn_outputs \
  --center-mode hybrid \
  --window-slices 64 \
  --image-size 128 \
  --base-ch 16 \
  --embed-dim 128 \
  --pooling attention \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda \
  --save-plots

python step8_train_lesion_models.py \
  --mode lesion_cnn \
  --data-root ./26S_AI536_NE450 \
  --out-dir step8_1b_lesion_cnn_multi_outputs \
  --center-mode multi \
  --multi-centers 3 \
  --window-slices 32 \
  --image-size 128 \
  --base-ch 16 \
  --embed-dim 128 \
  --pooling attention \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda \
  --save-plots

# step8-2
python step8_train_lesion_models.py \
  --mode fusion_cnn_tabular \
  --data-root ./26S_AI536_NE450 \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step8_2_lesion_fusion_outputs \
  --center-mode hybrid \
  --window-slices 64 \
  --image-size 128 \
  --base-ch 16 \
  --embed-dim 128 \
  --tab-embed-dim 128 \
  --pooling attention \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda \
  --save-plots

python step8_train_lesion_models.py \
  --mode fusion_cnn_tabular \
  --data-root ./26S_AI536_NE450 \
  --train-features step7_0_feature_v2_outputs/train_features_v2_merged.csv \
  --test-features step7_0_feature_v2_outputs/test_public_features_v2_merged.csv \
  --out-dir step8_2b_lesion_fusion_v2_outputs \
  --center-mode hybrid \
  --window-slices 64 \
  --image-size 128 \
  --base-ch 16 \
  --embed-dim 128 \
  --tab-embed-dim 128 \
  --pooling attention \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 100 \
  --patience 20 \
  --augment \
  --device cuda \
  --save-plots

# step8-3
python step8_train_lesion_models.py \
  --mode ffr_regression \
  --train-features step2_outputs/train_features.csv \
  --test-features step2_outputs/test_public_features.csv \
  --out-dir step8_3_ffr_regression_outputs \
  --hidden-dims 512 256 128 \
  --batch-size 64 \
  --eval-batch-size 256 \
  --epochs 300 \
  --patience 40 \
  --lr 0.0007 \
  --weight-decay 0.0001 \
  --device cuda \
  --save-plots

python step8_train_lesion_models.py \
  --mode ffr_regression \
  --train-features step7_0_feature_v2_outputs/train_features_v2_merged.csv \
  --test-features step7_0_feature_v2_outputs/test_public_features_v2_merged.csv \
  --out-dir step8_3b_ffr_regression_v2_outputs \
  --hidden-dims 512 256 128 \
  --batch-size 64 \
  --eval-batch-size 256 \
  --epochs 300 \
  --patience 40 \
  --lr 0.0007 \
  --weight-decay 0.0001 \
  --device cuda \
  --save-plots