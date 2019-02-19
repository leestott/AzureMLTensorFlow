#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks\python'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Exercise07 : Hyperparameter Tuning
# 
# AML provides framework-independent hyperparameter tuning capability.    
# This capability monitors accuracy in AML logs.
# 
# *back to [index](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/)*
#%% [markdown]
# ## Save your training code
# 
# First, you must save your training code.    
# Here we should use the source code in "[Exercise06 : Experimentation Logs and Outputs](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/blob/master/notebooks/exercise06_experimentation.ipynb)", which sends logs periodically into AML run history.
#%% [markdown]
# Create ```scirpt``` directory.

#%%
import os
script_folder = './script'
os.makedirs(script_folder, exist_ok=True)

#%% [markdown]
# Save source code as ```./script/train_expriment.py```.

#%%
get_ipython().run_cell_magic(u'writefile', u'script/train_experiment.py', u"from __future__ import absolute_import\nfrom __future__ import division\nfrom __future__ import print_function\n\nimport sys\nimport os\nimport shutil\nimport argparse\nimport math\n\nimport tensorflow as tf\n\nfrom azureml.core.run import Run\n\n# Get run when running in remote\nif 'run' not in locals():\n    run = Run.get_context()\n\nFLAGS = None\nbatch_size = 100\n\n#\n# define functions for Estimator\n#\n\ndef _my_input_fn(filepath, num_epochs):\n    # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]\n    # label - digit (0, 1, ..., 9)\n    data_queue = tf.train.string_input_producer(\n        [filepath],\n        num_epochs = num_epochs) # data is repeated and it raises OutOfRange when data is over\n    data_reader = tf.TFRecordReader()\n    _, serialized_exam = data_reader.read(data_queue)\n    data_exam = tf.parse_single_example(\n        serialized_exam,\n        features={\n            'image_raw': tf.FixedLenFeature([], tf.string),\n            'label': tf.FixedLenFeature([], tf.int64)\n        })\n    data_image = tf.decode_raw(data_exam['image_raw'], tf.uint8)\n    data_image.set_shape([784])\n    data_image = tf.cast(data_image, tf.float32) * (1. / 255)\n    data_label = tf.cast(data_exam['label'], tf.int32)\n    data_batch_image, data_batch_label = tf.train.batch(\n        [data_image, data_label],\n        batch_size=batch_size)\n    return {'inputs': data_batch_image}, data_batch_label\n\ndef _get_input_fn(filepath, num_epochs):\n    return lambda: _my_input_fn(filepath, num_epochs)\n\ndef _my_model_fn(features, labels, mode):\n    # with tf.device(...): # You can set device if using GPUs\n\n    # define network and inference\n    # (simple 2 fully connected hidden layer : 784->128->64->10)\n    with tf.name_scope('hidden1'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [784, FLAGS.first_layer],\n                stddev=1.0 / math.sqrt(float(784))),\n            name='weights')\n        biases = tf.Variable(\n            tf.zeros([FLAGS.first_layer]),\n            name='biases')\n        hidden1 = tf.nn.relu(tf.matmul(features['inputs'], weights) + biases)\n    with tf.name_scope('hidden2'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [FLAGS.first_layer, FLAGS.second_layer],\n                stddev=1.0 / math.sqrt(float(FLAGS.first_layer))),\n            name='weights')\n        biases = tf.Variable(\n            tf.zeros([FLAGS.second_layer]),\n            name='biases')\n        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)\n    with tf.name_scope('softmax_linear'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [FLAGS.second_layer, 10],\n                stddev=1.0 / math.sqrt(float(FLAGS.second_layer))),\n        name='weights')\n        biases = tf.Variable(\n            tf.zeros([10]),\n            name='biases')\n        logits = tf.matmul(hidden2, weights) + biases\n \n    # compute evaluation matrix\n    predicted_indices = tf.argmax(input=logits, axis=1)\n    if mode != tf.estimator.ModeKeys.PREDICT:\n        label_indices = tf.cast(labels, tf.int32)\n        accuracy = tf.metrics.accuracy(label_indices, predicted_indices)\n        tf.summary.scalar('accuracy', accuracy[1]) # output to TensorBoard \n        loss = tf.losses.sparse_softmax_cross_entropy(\n            labels=labels,\n            logits=logits)\n \n    # define operations\n    if mode == tf.estimator.ModeKeys.TRAIN:\n        #global_step = tf.train.create_global_step()\n        #global_step = tf.contrib.framework.get_or_create_global_step()\n        global_step = tf.train.get_or_create_global_step()        \n        optimizer = tf.train.GradientDescentOptimizer(\n            learning_rate=FLAGS.learning_rate)\n        train_op = optimizer.minimize(\n            loss=loss,\n            global_step=global_step)\n        # Ask for accuracy and loss in each steps\n        class _CustomLoggingHook(tf.train.SessionRunHook):\n            def begin(self):\n                self.training_accuracy = []\n                self.training_loss = []\n            def before_run(self, run_context):\n                return tf.train.SessionRunArgs([accuracy[1], loss, global_step])\n            def after_run(self, run_context, run_values):\n                result_accuracy, result_loss, result_step = run_values.results\n                if result_step % 10 == 0 :\n                    self.training_accuracy.append(result_accuracy)\n                    self.training_loss.append(result_loss)\n                if result_step % 100 == 0 : # save logs in each 100 steps\n                    run.log_list('training_accuracy', self.training_accuracy)\n                    run.log_list('training_loss', self.training_loss)\n                    self.training_accuracy = []\n                    self.training_loss = []\n        return tf.estimator.EstimatorSpec(\n            mode,\n            training_chief_hooks=[_CustomLoggingHook()],\n            loss=loss,\n            train_op=train_op)\n    if mode == tf.estimator.ModeKeys.EVAL:\n        eval_metric_ops = {\n            'accuracy': accuracy\n        }\n        return tf.estimator.EstimatorSpec(\n            mode,\n            loss=loss,\n            eval_metric_ops=eval_metric_ops)\n    if mode == tf.estimator.ModeKeys.PREDICT:\n        probabilities = tf.nn.softmax(logits, name='softmax_tensor')\n        predictions = {\n            'classes': predicted_indices,\n            'probabilities': probabilities\n        }\n        export_outputs = {\n            'prediction': tf.estimator.export.PredictOutput(predictions)\n        }\n        return tf.estimator.EstimatorSpec(\n            mode,\n            predictions=predictions,\n            export_outputs=export_outputs)\n\ndef _my_serving_input_fn():\n    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}\n    return tf.estimator.export.ServingInputReceiver(inputs, inputs)\n\n#\n# Main\n#\n\nparser = argparse.ArgumentParser()\nparser.add_argument(\n    '--data_folder',\n    type=str,\n    default='./data',\n    help='Folder path for input data')\nparser.add_argument(\n    '--chkpoint_folder',\n    type=str,\n    default='./logs',  # AML experiments logs folder\n    help='Folder path for checkpoint files')\nparser.add_argument(\n    '--model_folder',\n    type=str,\n    default='./outputs',  # AML experiments outputs folder\n    help='Folder path for model output')\nparser.add_argument(\n    '--learning_rate',\n    type=float,\n    default='0.07',\n    help='Learning Rate')\nparser.add_argument(\n    '--first_layer',\n    type=int,\n    default='128',\n    help='Neuron number for the first hidden layer')\nparser.add_argument(\n    '--second_layer',\n    type=int,\n    default='64',\n    help='Neuron number for the second hidden layer')\nFLAGS, unparsed = parser.parse_known_args()\n\n# clean checkpoint and model folder if exists\nif os.path.exists(FLAGS.chkpoint_folder) :\n    for file_name in os.listdir(FLAGS.chkpoint_folder):\n        file_path = os.path.join(FLAGS.chkpoint_folder, file_name)\n        if os.path.isfile(file_path):\n            os.remove(file_path)\n        elif os.path.isdir(file_path):\n            shutil.rmtree(file_path)\nif os.path.exists(FLAGS.model_folder) :\n    for file_name in os.listdir(FLAGS.model_folder):\n        file_path = os.path.join(FLAGS.model_folder, file_name)\n        if os.path.isfile(file_path):\n            os.remove(file_path)\n        elif os.path.isdir(file_path):\n            shutil.rmtree(file_path)\n\n# read TF_CONFIG\nrun_config = tf.contrib.learn.RunConfig()\n\n# create Estimator\nmnist_fullyconnected_classifier = tf.estimator.Estimator(\n    model_fn=_my_model_fn,\n    model_dir=FLAGS.chkpoint_folder,\n    config=run_config)\ntrain_spec = tf.estimator.TrainSpec(\n    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'train.tfrecords'), 2),\n    max_steps=60000 * 2 / batch_size)\neval_spec = tf.estimator.EvalSpec(\n    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'test.tfrecords'), 1),\n    steps=10000 * 1 / batch_size,\n    start_delay_secs=0)\n\n# run !\neval_res = tf.estimator.train_and_evaluate(\n    mnist_fullyconnected_classifier,\n    train_spec,\n    eval_spec\n)\n\n# save model and variables\nmodel_dir = mnist_fullyconnected_classifier.export_savedmodel(\n    export_dir_base = FLAGS.model_folder,\n    serving_input_receiver_fn = _my_serving_input_fn)\nprint('current working directory is ', os.getcwd())\nprint('model is saved ', model_dir)\n\n# send logs to AML\nrun.log('learning_rate', FLAGS.learning_rate)\nrun.log('1st_layer', FLAGS.first_layer)\nrun.log('2nd_layer', FLAGS.second_layer)\nrun.log('final_accuracy', eval_res[0]['accuracy'])\nrun.log('final_loss', eval_res[0]['loss'])")

#%% [markdown]
# ## Get workspace setting
# 
# Before starting, you must read your configuration settings. (See "[Exercise01 : Prepare Config Settings](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/blob/master/notebooks/exercise01_prepare_config.ipynb)".)

#%%
from azureml.core import Workspace
import azureml.core

ws = Workspace.from_config()

#%% [markdown]
# ## Create AML compute
# 
# Create AML compute pool for computing environment.

#%%
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

try:
    compute_target = ComputeTarget(workspace=ws, name='hypertest01')
    print('found existing:', compute_target.name)
except ComputeTargetException:
    print('creating new.')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_D2_v2',
        min_nodes=0,
        max_nodes=4)
    compute_target = ComputeTarget.create(ws, 'hypertest01', compute_config)
    compute_target.wait_for_completion(show_output=True)


#%%
# get a status for the current cluster.
print(compute_target.status.serialize())

#%% [markdown]
# ## Prepare Datastore
# 
# You can mount your datastore (See "[Exercise02 : Prepare Datastore](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/blob/master/notebooks/exercise02_prepare_datastore.ipynb)") into your Batch AI compute.

#%%
from azureml.core import Datastore

# get your datastore (See "Exercise 02 : Prepare Datastore")
ds = Datastore.get(ws, datastore_name="myblob01")
ds_data = ds.path('tfdata')

#%% [markdown]
# ## Generate Hyperparameter Sampling
# 
# Set how to explorer for script (```train_experiment.py```) parameters.    
# You can choose from ```GridParameterSampling```, ```RandomParameterSampling```, and ```BayesianParameterSampling```.

#%%
from azureml.train.hyperdrive import *

param_sampling = RandomParameterSampling(
    {
        '--learning_rate': choice(0.01, 0.05, 0.9),
        '--first_layer': choice(100, 125, 150),
        '--second_layer': choice(30, 60, 90)
    }
)

#%% [markdown]
# ## Generate estimator

#%%
from azureml.train.dnn import TensorFlow

script_params={
    '--data_folder': ds_data
}
estimator = TensorFlow(
    source_directory='./script',
    compute_target=compute_target,
    script_params=script_params,
    entry_script='train_experiment.py',
    use_gpu=False)

#%% [markdown]
# ## Generate run config
# 
# Generate run config with an early termnination policy (```BanditPolicy```). With this policy, the training will terminate if the primary metric falls outside of the top 10% range (checking every 2 iterations).

#%%
# early termnination :
# primary metric falls outside of the top 10% (0.1) range by checking every 2 iterations
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
# generate run config
run_config = HyperDriveRunConfig(
    estimator=estimator,
    hyperparameter_sampling=param_sampling,
    primary_metric_name='training_accuracy',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
    policy=policy,
    max_total_runs=20,
    max_concurrent_runs=4)

#%% [markdown]
# ## Run script and wait for completion

#%%
from azureml.core import Experiment

experiment = Experiment(workspace=ws, name='hyperdrive_test')
run = experiment.submit(config=run_config)


#%%
run.wait_for_completion(show_output=True)

#%% [markdown]
# ## View logs
#%% [markdown]
# You can view logs using [Azure Portal](https://portal.azure.com/), but you can also view using AML run history widget in your notebook.

#%%
from azureml.widgets import RunDetails
RunDetails(run_instance=run).show()

#%% [markdown]
# You can also explorer metrics with your python code.

#%%
allmetrics = run.get_metrics()
print(allmetrics)

#%% [markdown]
# ## Remove AML compute

#%%
# Delete cluster (nbodes) and remove from AML workspace
mycompute = AmlCompute(workspace=ws, name='hypertest01')
mycompute.delete()


#%%



