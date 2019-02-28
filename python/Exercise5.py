#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Exercise05 : Distributed Training
# 
# Here we change our sample (see "[Exercise03 : Just Train in Your Working Machine](/notebooks/exercise03_train_simple.ipynb)") for distributed training using multiple machines.
# 
# In this exercise we use Horovod framework (https://github.com/horovod/horovod) using built-in ```azureml.train.dnn.TensorFlow``` estimator, but you can also configure using primitive ```azureml.core.ScriptRunConfig``` for the training with TensorFlow and Horovod. (See [here](https://tsmatz.wordpress.com/2019/01/17/azure-machine-learning-service-custom-amlcompute-and-runconfig-for-mxnet-distributed-training/) for sample script with TensorFlow and Horovod using ```azureml.core.ScriptRunConfig```.)
# 
# *back to [index](/Readme.md)*
#%% [markdown]
# ## Save your training script as file (train.py)
#%% [markdown]
# Create ```scirpt``` directory.

#%%
import os
script_folder = './script'
os.makedirs(script_folder, exist_ok=True)

#%% [markdown]
# Change our original source code ```train.py``` (see "[Exercise03 : Just Train in Your Working Machine](/notebooks/exercise03_train_simple.ipynb)") as follows. The lines commented "##### modified" is modified lines.    
# After that, please add the following ```%%writefile``` at the beginning of the source code and run this cell.    
# This source code is saved as ```./script/train_horovod.py```.

#%%
get_ipython().run_cell_magic(u'writefile', u'script/train_horovod.py', u"from __future__ import absolute_import\nfrom __future__ import division\nfrom __future__ import print_function\n\nimport sys\nimport os\nimport shutil\nimport argparse\nimport math\n\nimport tensorflow as tf\nimport horovod.tensorflow as hvd ##### modified\n\nFLAGS = None\nbatch_size = 100\n\n#\n# define functions for Estimator\n#\n\ndef _my_input_fn(filepath, num_epochs):\n    # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]\n    # label - digit (0, 1, ..., 9)\n    data_queue = tf.train.string_input_producer(\n        [filepath],\n        num_epochs = num_epochs) # data is repeated and it raises OutOfRange when data is over\n    data_reader = tf.TFRecordReader()\n    _, serialized_exam = data_reader.read(data_queue)\n    data_exam = tf.parse_single_example(\n        serialized_exam,\n        features={\n            'image_raw': tf.FixedLenFeature([], tf.string),\n            'label': tf.FixedLenFeature([], tf.int64)\n        })\n    data_image = tf.decode_raw(data_exam['image_raw'], tf.uint8)\n    data_image.set_shape([784])\n    data_image = tf.cast(data_image, tf.float32) * (1. / 255)\n    data_label = tf.cast(data_exam['label'], tf.int32)\n    data_batch_image, data_batch_label = tf.train.batch(\n        [data_image, data_label],\n        batch_size=batch_size)\n    return {'inputs': data_batch_image}, data_batch_label\n\ndef _get_input_fn(filepath, num_epochs):\n    return lambda: _my_input_fn(filepath, num_epochs)\n\ndef _my_model_fn(features, labels, mode):\n    # with tf.device(...): # You can set device if using GPUs\n\n    # define network and inference\n    # (simple 2 fully connected hidden layer : 784->128->64->10)\n    with tf.name_scope('hidden1'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [784, FLAGS.first_layer],\n                stddev=1.0 / math.sqrt(float(784))),\n            name='weights')\n        biases = tf.Variable(\n            tf.zeros([FLAGS.first_layer]),\n            name='biases')\n        hidden1 = tf.nn.relu(tf.matmul(features['inputs'], weights) + biases)\n    with tf.name_scope('hidden2'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [FLAGS.first_layer, FLAGS.second_layer],\n                stddev=1.0 / math.sqrt(float(FLAGS.first_layer))),\n            name='weights')\n        biases = tf.Variable(\n            tf.zeros([FLAGS.second_layer]),\n            name='biases')\n        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)\n    with tf.name_scope('softmax_linear'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [FLAGS.second_layer, 10],\n                stddev=1.0 / math.sqrt(float(FLAGS.second_layer))),\n        name='weights')\n        biases = tf.Variable(\n            tf.zeros([10]),\n            name='biases')\n        logits = tf.matmul(hidden2, weights) + biases\n \n    # compute evaluation matrix\n    predicted_indices = tf.argmax(input=logits, axis=1)\n    if mode != tf.estimator.ModeKeys.PREDICT:\n        label_indices = tf.cast(labels, tf.int32)\n        accuracy = tf.metrics.accuracy(label_indices, predicted_indices)\n        tf.summary.scalar('accuracy', accuracy[1]) # output to TensorBoard\n \n        loss = tf.losses.sparse_softmax_cross_entropy(\n            labels=labels,\n            logits=logits)\n \n    # define operations\n    if mode == tf.estimator.ModeKeys.TRAIN:\n        global_step = tf.train.get_or_create_global_step()        \n        optimizer = tf.train.GradientDescentOptimizer(\n            learning_rate=FLAGS.learning_rate)\n        optimizer = hvd.DistributedOptimizer(optimizer) ##### modified\n        train_op = optimizer.minimize(\n            loss=loss,\n            global_step=global_step)\n        return tf.estimator.EstimatorSpec(\n            mode,\n            loss=loss,\n            train_op=train_op)\n    if mode == tf.estimator.ModeKeys.EVAL:\n        eval_metric_ops = {\n            'accuracy': accuracy\n        }\n        return tf.estimator.EstimatorSpec(\n            mode,\n            loss=loss,\n            eval_metric_ops=eval_metric_ops)\n    if mode == tf.estimator.ModeKeys.PREDICT:\n        probabilities = tf.nn.softmax(logits, name='softmax_tensor')\n        predictions = {\n            'classes': predicted_indices,\n            'probabilities': probabilities\n        }\n        export_outputs = {\n            'prediction': tf.estimator.export.PredictOutput(predictions)\n        }\n        return tf.estimator.EstimatorSpec(\n            mode,\n            predictions=predictions,\n            export_outputs=export_outputs)\n\ndef _my_serving_input_fn():\n    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}\n    return tf.estimator.export.ServingInputReceiver(inputs, inputs)\n\n#\n# Main\n#\n\nparser = argparse.ArgumentParser()\nparser.add_argument(\n    '--data_folder',\n    type=str,\n    default='./data',\n    help='Folder path for input data')\nparser.add_argument(\n    '--chkpoint_folder',\n    type=str,\n    default='./logs',  # AML experiments logs folder\n    help='Folder path for checkpoint files')\nparser.add_argument(\n    '--model_folder',\n    type=str,\n    default='./outputs',  # AML experiments outputs folder\n    help='Folder path for model output')\nparser.add_argument(\n    '--learning_rate',\n    type=float,\n    default='0.07',\n    help='Learning Rate')\nparser.add_argument(\n    '--first_layer',\n    type=int,\n    default='128',\n    help='Neuron number for the first hidden layer')\nparser.add_argument(\n    '--second_layer',\n    type=int,\n    default='64',\n    help='Neuron number for the second hidden layer')\nFLAGS, unparsed = parser.parse_known_args()\n\n# clean checkpoint and model folder if exists\nif os.path.exists(FLAGS.chkpoint_folder) :\n    for file_name in os.listdir(FLAGS.chkpoint_folder):\n        file_path = os.path.join(FLAGS.chkpoint_folder, file_name)\n        if os.path.isfile(file_path):\n            os.remove(file_path)\n        elif os.path.isdir(file_path):\n            shutil.rmtree(file_path)\nif os.path.exists(FLAGS.model_folder) :\n    for file_name in os.listdir(FLAGS.model_folder):\n        file_path = os.path.join(FLAGS.model_folder, file_name)\n        if os.path.isfile(file_path):\n            os.remove(file_path)\n        elif os.path.isdir(file_path):\n            shutil.rmtree(file_path)\n\nhvd.init() ##### modified\n\n# read TF_CONFIG\nrun_config = tf.contrib.learn.RunConfig()\n\n# create Estimator\nmnist_fullyconnected_classifier = tf.estimator.Estimator(\n    model_fn=_my_model_fn,\n    model_dir=FLAGS.chkpoint_folder if hvd.rank() == 0 else None, ##### modified\n    config=run_config)\ntrain_spec = tf.estimator.TrainSpec(\n    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'train.tfrecords'), 2),\n    #max_steps=60000 * 2 / batch_size)\n    max_steps=(60000 * 2 / batch_size) // hvd.size(), ##### modified\n    hooks=[hvd.BroadcastGlobalVariablesHook(0)]) ##### modified\neval_spec = tf.estimator.EvalSpec(\n    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'test.tfrecords'), 1),\n    steps=10000 * 1 / batch_size,\n    start_delay_secs=0)\n\n# run !\ntf.estimator.train_and_evaluate(\n    mnist_fullyconnected_classifier,\n    train_spec,\n    eval_spec\n)\n\n# save model and variables\nif hvd.rank() == 0 : ##### modified\n    model_dir = mnist_fullyconnected_classifier.export_savedmodel(\n        export_dir_base = FLAGS.model_folder,\n        serving_input_receiver_fn = _my_serving_input_fn)\n    print('current working directory is ', os.getcwd())\n    print('model is saved ', model_dir)")

#%% [markdown]
# ## Train on multiple machines (Horovod)
#%% [markdown]
# ### Step 1 : Get workspace setting
# 
# Before starting, you must read your configuration settings. (See "[Exercise01 : Prepare Config Settings](/notebooks/exercise01_prepare_config.ipynb)".)

#%%
from azureml.core import Workspace
import azureml.core

ws = Workspace.from_config()

#%% [markdown]
# ### Step 2 : Create multiple virtual machines (cluster)
# 
# Create your new AML compute for distributed clusters. By enabling auto-scaling from 0 to 4, you can save money (all nodes are terminated) if it's inactive. see https://docs.microsoft.com/en-us/azure/architecture/best-practices/auto-scaling 
# If already exists, this script will get the existing cluster. The script below creates a cluster of  D2_v2 machines - vm_size='Standard_D2_v2',

#%%
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

try:
    compute_target = ComputeTarget(workspace=ws, name='mycluster01')
    print('found existing:', compute_target.name)
except ComputeTargetException:
    print('creating new.')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_D2_v2',
        min_nodes=0,
        max_nodes=4)
    compute_target = ComputeTarget.create(ws, 'mycluster01', compute_config)
    compute_target.wait_for_completion(show_output=True)


#%%
# get a status for the current cluster.
print(compute_target.status.serialize())

#%% [markdown]
# ### Step 3 : Prepare datastore
# 
# You can mount your datastore (See "[Exercise02 : Prepare Datastore](/notebooks/exercise02_prepare_datastore.ipynb)") into your Batch AI compute.

#%%
from azureml.core import Datastore

# get your datastore (See "Exercise 02 : Prepare Datastore")
ds = Datastore.get(ws, datastore_name="myblob01")
ds_data = ds.path('tfdata')

#%% [markdown]
# ### Step 4 : Generate estimator **
# 
# Run distributed training by Horovod using built-in ```azureml.train.dnn.TensorFlow``` estimator.    
# If you want to customize more detailed settings (other frameworks, custom images, etc), please use base ```azureml.train.estimator.Estimator``` (parent class).
# 
# ** Note : This estimator (```azureml.train.dnn.TensorFlow```) is an estimator in AML SDK, and not the same as ```tf.estimator.Estimator``` in TensorFlow. Do not confused for the terminology "Estimator".

#%%
from azureml.train.dnn import TensorFlow

script_params={
    '--data_folder': ds_data
}
estimator = TensorFlow(
    source_directory='./script',
    compute_target=compute_target,
    script_params=script_params,
    entry_script='train_horovod.py',
    node_count=2,
    process_count_per_node=1,
    distributed_backend='mpi',
    use_gpu=False)

#%% [markdown]
# ### Step 5 : Run script and wait for completion

#%%
from azureml.core import Experiment

exp = Experiment(workspace=ws, name='tf_distribued')
run = exp.submit(estimator)
run.wait_for_completion(show_output=True)

#%% [markdown]
# ### Step 6 : Check results

#%%
run.get_file_names()

#%% [markdown]
# **Please change ```1544487483``` to meet previous results.**

#%%
run.download_file(
    name='outputs/1544487483/saved_model.pb',
    output_file_path='distributed_model/saved_model.pb')
run.download_file(
    name='outputs/1544487483/variables/variables.data-00000-of-00001',
    output_file_path='distributed_model/variables/variables.data-00000-of-00001')
run.download_file(
    name='outputs/1544487483/variables/variables.index',
    output_file_path='distributed_model/variables/variables.index')


#%%
import tensorflow as tf

# Read data by tensor
dataset = tf.data.TFRecordDataset('./data/test.tfrecords')
iterator = dataset.make_one_shot_iterator()
data_org = iterator.get_next()
data_exam = tf.parse_single_example(
    data_org,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
data_image = tf.decode_raw(data_exam['image_raw'], tf.uint8)
data_image.set_shape([784])
data_image = tf.cast(data_image, tf.float32) * (1. / 255)
data_label = tf.cast(data_exam['label'], tf.int32)

# Run tensor and generate data
with tf.Session() as sess:
    image_arr = []
    label_arr = []
    for i in range(3):
        image, label = sess.run([data_image, data_label])
        image_arr.append(image)
        label_arr.append(label)

# Predict
pred_fn = tf.contrib.predictor.from_saved_model('./distributed_model')
pred = pred_fn({'inputs': image_arr})

print('Predicted: ', pred['classes'].tolist())
print('Actual   : ', label_arr)

#%% [markdown]
# ### Step 6 : Remove AML compute
# 
# **You don't need to remove your AML compute** for saving money, because the nodes will be automatically terminated, when it's inactive.    
# But if you want to clean up, please run the following.

#%%
# Delete cluster (nbodes) and remove from AML workspace
mycompute = AmlCompute(workspace=ws, name='mycluster01')
mycompute.delete()


#%%



