# load raw data

### load and transform data for train

## Inputs

`file_name` input raw file name, String type.

`data_dir` input data path, String type.

## Outputs

`x_train_path` train feature data path, String type.

`x_test_path` test feature data path, String type.

`y_train_path` train label data path, String type.

`t_test_path` train label data path, String type.

## Container image

kubeflow/mnist-load_data:v0.0.4

## Usage

load raw data, split data for train and test.



