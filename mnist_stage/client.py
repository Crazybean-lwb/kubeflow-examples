import kfp
from kfp import compiler
import kfp.dsl as dsl
import kfp.notebook
import kfp.gcp as gcp

client = kfp.Client()
from kubernetes import client as k8s_client

EXPERIMENT_NAME = 'mnist_op'
exp = client.create_experiment(name=EXPERIMENT_NAME)


class load_dataOp(dsl.ContainerOp):
    """load raw data from tensorflow, do data transform"""

    def __init__(self, data_dir, file_name):
        super(load_dataOp, self).__init__(
            name='load_data',
            image='mnist-load_data:v0.0.1',
            arguments=[
                '--file_name', file_name,
                '--data_dir', data_dir,
            ],
            file_outputs={
                'data_file': data_dir + 'train_test_data.txt'
            })


class trainOp(dsl.ContainerOp):
    """train keras model"""

    def __init__(self, data_dir, data_file):
        super(trainOp, self).__init__(
            name='train',
            image='mnist-train:v0.0.1',
            arguments=[
                '--data_dir', data_dir,
                '--data_file', data_file,
            ],
            file_outputs={
                'model_file': data_dir + 'model.txt'
            })


class predictOp(dsl.ContainerOp):
    """get predict by trained model"""

    def __init__(self, data_dir, model_file, data_file):
        super(predictOp, self).__init__(
            name='predict',
            image='mnist-predict:v0.0.1',
            arguments=[
                '--data_dir', data_dir,
                '--model_file', model_file,
                '--data_file', data_file
            ],
            file_outputs={
                'result_file': data_dir + 'result.txt'
            })


@dsl.pipeline(
    name='MnistStage',
    description='shows how to define dsl.Condition.'
)
def MnistTest():
    data_dir = '/DATA/nfs-data/kubeflow-pv1/'
    file_name = 'mnist.npz'
    load_data = load_dataOp(data_dir, file_name).add_volume(k8s_client.V1Volume(name='mnist-pv',
                                                                                nfs=k8s_client.V1NFSVolumeSource(
                                                                                    path='/DATA/nfs-data/kubeflow-pv1/',
                                                                                    server='10.5.188.249'))).add_volume_mount(
        k8s_client.V1VolumeMount(mount_path='/DATA/nfs-data/kubeflow-pv1/', name='mnist-pv'))

    train = trainOp(data_dir, load_data.outputs['data_file']).add_volume(k8s_client.V1Volume(name='mnist-pv',
                                                                                             nfs=k8s_client.V1NFSVolumeSource(
                                                                                                 path='/DATA/nfs-data/kubeflow-pv1/',
                                                                                                 server='10.5.188.249'))).add_volume_mount(
        k8s_client.V1VolumeMount(mount_path='/DATA/nfs-data/kubeflow-pv1/', name='mnist-pv'))

    predict = predictOp(data_dir, train.outputs['model_file'], load_data.outputs['data_file']).add_volume(
        k8s_client.V1Volume(name='mnist-pv',
                            nfs=k8s_client.V1NFSVolumeSource(
                                path='/DATA/nfs-data/kubeflow-pv1/',
                                server='10.5.188.249'))).add_volume_mount(
        k8s_client.V1VolumeMount(mount_path='/DATA/nfs-data/kubeflow-pv1/', name='mnist-pv'))


compiler.Compiler().compile(MnistTest, 'mnist.tar.gz')
run = client.run_pipeline(exp.id, 'wbliu3', 'mnist.tar.gz')
