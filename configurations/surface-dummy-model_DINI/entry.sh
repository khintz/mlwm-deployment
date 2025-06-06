#!/usr/bin/env bash
# This script is used to run the inference for the surface-dummy-model_DINI model.
#
# This script is intended to be run in a container, and assumes that during the
# container image build that the inference artifact was unpacked to
# inference_artifact/


INFERENCE_ARTIFACT_PATH="./inference_artifact"
# XXX: these mount points could come from config.yaml for the model run configuration
INPUT_DATASETS_ROOT_PATH="/volume/inputs"
OUTPUT_DATASETS_ROOT_PATH="/volume/outputs"

# forecast out to 18 hours, which means 6 steps of 3 hours each (the model was
# trained on 3-hourly analysis data)
NUM_EVAL_STEPS=6

## 1. Create inference dataset
# This uses a cli stored within mlwm to called mllam-data-prep to create the
# inference dataset. The inference dataset is created by modifying the
# configuration used during training to a) change the paths to the input datasets,
# b) include the statistics from the training dataset and c) set the dimensions
# in the configuration to have `analysis_time` and `elapsed_forecast_duration`
# instead of just `time`.
uv run python -m mlwm.create_inference_dataset \
    --config_path ${INFERENCE_ARTIFACT_PATH}/config.yaml \
    --override_input_paths \
    danra_surface=${INPUT_DATASETS_ROOT_PATH}/single_levels.zarr \
    danra_surface_forcing=${INPUT_DATASETS_ROOT_PATH}/single_levels.zarr \
    danra_static=${INPUT_DATASETS_ROOT_PATH}/single_levels.zarr \
    --use_stats_from_path ${INFERENCE_ARTIFACT_PATH}/danra.datastore.stats.zarr \
    --output_root_path inference/

## 2. Create graph
uv run python -m neural_lam.create_graph --config_path inference/config.yaml

## 3. Run inference
uv run python -m neural_lam.train_model --config_path inference/config.yaml \
    --eval \
    --graph multiscale \
    --hidden_dim 2 \
    --ar_steps_eval ${NUM_EVAL_STEPS} \
    --load ${INFERENCE_ARTIFACT_PATH}/checkpoint.ckpt \
    --save_eval_to_zarr_path ${OUTPUT_DATASETS_ROOT_PATH}/inference_output.zarr

## 4. Transform inference output back to original grid and variables
# TODO: this will result in {input_name}.zarr dataset for each input that is
# used for constructing state-variables that the model should predict. This
# means that we will have `danra_surface.zarr` in this case. We rename name
# that manually here but maybe mllam-data-prep should be able to merge inputs
# originating from the same zarr dataset path?
uv run python -m mllam_data_prep.recreate_inputs ${OUTPUT_DATASETS_ROOT_PATH}/inference_output.zarr
rename ${OUTPUT_DATASETS_ROOT_PATH}/danra_surface.zarr ${OUTPUT_DATASETS_ROOT_PATH}/single_levels.zarr
