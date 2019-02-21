#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Exercise04 : Train on Remote GPU Virtual Machine
# 
# Now we run our previous sample (see "[Exercise03 : Just Train in Your Working Machine](/notebooks/exercise03_train_simple.ipynb)") on remote virtual machine with GPU utilized.    
# Here we use remote virtual machine and conda virtual environment, but you can also use Batch AI pool sharing in your team, or run on your favorite docker images.
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
# Please add the following ```%%writefile``` at the beginning of the source code in "[Exercise03 : Just Train in Your Working Machine](/notebooks/exercise03_train_simple.ipynb)", and run this cell.    
# Then this source code is saved as ```./script/train.py```.

#%%
get_ipython().run_cell_magic(u'writefile', u'script/train.py', u"from __future__ import absolute_import\nfrom __future__ import division\nfrom __future__ import print_function\n\nimport sys\nimport os\nimport shutil\nimport argparse\nimport math\n\nimport tensorflow as tf\n\nFLAGS = None\nbatch_size = 100\n\n#\n# define functions for Estimator\n#\n\ndef _my_input_fn(filepath, num_epochs):\n    # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]\n    # label - digit (0, 1, ..., 9)\n    data_queue = tf.train.string_input_producer(\n        [filepath],\n        num_epochs = num_epochs) # data is repeated and it raises OutOfRange when data is over\n    data_reader = tf.TFRecordReader()\n    _, serialized_exam = data_reader.read(data_queue)\n    data_exam = tf.parse_single_example(\n        serialized_exam,\n        features={\n            'image_raw': tf.FixedLenFeature([], tf.string),\n            'label': tf.FixedLenFeature([], tf.int64)\n        })\n    data_image = tf.decode_raw(data_exam['image_raw'], tf.uint8)\n    data_image.set_shape([784])\n    data_image = tf.cast(data_image, tf.float32) * (1. / 255)\n    data_label = tf.cast(data_exam['label'], tf.int32)\n    data_batch_image, data_batch_label = tf.train.batch(\n        [data_image, data_label],\n        batch_size=batch_size)\n    return {'inputs': data_batch_image}, data_batch_label\n\ndef _get_input_fn(filepath, num_epochs):\n    return lambda: _my_input_fn(filepath, num_epochs)\n\ndef _my_model_fn(features, labels, mode):\n    # with tf.device(...): # You can set device if using GPUs\n\n    # define network and inference\n    # (simple 2 fully connected hidden layer : 784->128->64->10)\n    with tf.name_scope('hidden1'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [784, FLAGS.first_layer],\n                stddev=1.0 / math.sqrt(float(784))),\n            name='weights')\n        biases = tf.Variable(\n            tf.zeros([FLAGS.first_layer]),\n            name='biases')\n        hidden1 = tf.nn.relu(tf.matmul(features['inputs'], weights) + biases)\n    with tf.name_scope('hidden2'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [FLAGS.first_layer, FLAGS.second_layer],\n                stddev=1.0 / math.sqrt(float(FLAGS.first_layer))),\n            name='weights')\n        biases = tf.Variable(\n            tf.zeros([FLAGS.second_layer]),\n            name='biases')\n        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)\n    with tf.name_scope('softmax_linear'):\n        weights = tf.Variable(\n            tf.truncated_normal(\n                [FLAGS.second_layer, 10],\n                stddev=1.0 / math.sqrt(float(FLAGS.second_layer))),\n        name='weights')\n        biases = tf.Variable(\n            tf.zeros([10]),\n            name='biases')\n        logits = tf.matmul(hidden2, weights) + biases\n \n    # compute evaluation matrix\n    predicted_indices = tf.argmax(input=logits, axis=1)\n    if mode != tf.estimator.ModeKeys.PREDICT:\n        label_indices = tf.cast(labels, tf.int32)\n        accuracy = tf.metrics.accuracy(label_indices, predicted_indices)\n        tf.summary.scalar('accuracy', accuracy[1]) # output to TensorBoard\n \n        loss = tf.losses.sparse_softmax_cross_entropy(\n            labels=labels,\n            logits=logits)\n \n    # define operations\n    if mode == tf.estimator.ModeKeys.TRAIN:\n        #global_step = tf.train.create_global_step()\n        #global_step = tf.contrib.framework.get_or_create_global_step()\n        global_step = tf.train.get_or_create_global_step()        \n        optimizer = tf.train.GradientDescentOptimizer(\n            learning_rate=FLAGS.learning_rate)\n        train_op = optimizer.minimize(\n            loss=loss,\n            global_step=global_step)\n        return tf.estimator.EstimatorSpec(\n            mode,\n            loss=loss,\n            train_op=train_op)\n    if mode == tf.estimator.ModeKeys.EVAL:\n        eval_metric_ops = {\n            'accuracy': accuracy\n        }\n        return tf.estimator.EstimatorSpec(\n            mode,\n            loss=loss,\n            eval_metric_ops=eval_metric_ops)\n    if mode == tf.estimator.ModeKeys.PREDICT:\n        probabilities = tf.nn.softmax(logits, name='softmax_tensor')\n        predictions = {\n            'classes': predicted_indices,\n            'probabilities': probabilities\n        }\n        export_outputs = {\n            'prediction': tf.estimator.export.PredictOutput(predictions)\n        }\n        return tf.estimator.EstimatorSpec(\n            mode,\n            predictions=predictions,\n            export_outputs=export_outputs)\n\ndef _my_serving_input_fn():\n    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}\n    return tf.estimator.export.ServingInputReceiver(inputs, inputs)\n\n#\n# Main\n#\n\nparser = argparse.ArgumentParser()\nparser.add_argument(\n    '--data_folder',\n    type=str,\n    default='./data',\n    help='Folder path for input data')\nparser.add_argument(\n    '--chkpoint_folder',\n    type=str,\n    default='./logs',  # AML experiments logs folder\n    help='Folder path for checkpoint files')\nparser.add_argument(\n    '--model_folder',\n    type=str,\n    default='./outputs',  # AML experiments outputs folder\n    help='Folder path for model output')\nparser.add_argument(\n    '--learning_rate',\n    type=float,\n    default='0.07',\n    help='Learning Rate')\nparser.add_argument(\n    '--first_layer',\n    type=int,\n    default='128',\n    help='Neuron number for the first hidden layer')\nparser.add_argument(\n    '--second_layer',\n    type=int,\n    default='64',\n    help='Neuron number for the second hidden layer')\nFLAGS, unparsed = parser.parse_known_args()\n\n# clean checkpoint and model folder if exists\nif os.path.exists(FLAGS.chkpoint_folder) :\n    for file_name in os.listdir(FLAGS.chkpoint_folder):\n        file_path = os.path.join(FLAGS.chkpoint_folder, file_name)\n        if os.path.isfile(file_path):\n            os.remove(file_path)\n        elif os.path.isdir(file_path):\n            shutil.rmtree(file_path)\nif os.path.exists(FLAGS.model_folder) :\n    for file_name in os.listdir(FLAGS.model_folder):\n        file_path = os.path.join(FLAGS.model_folder, file_name)\n        if os.path.isfile(file_path):\n            os.remove(file_path)\n        elif os.path.isdir(file_path):\n            shutil.rmtree(file_path)\n\n# read TF_CONFIG\nrun_config = tf.contrib.learn.RunConfig()\n\n# create Estimator\nmnist_fullyconnected_classifier = tf.estimator.Estimator(\n    model_fn=_my_model_fn,\n    model_dir=FLAGS.chkpoint_folder,\n    config=run_config)\ntrain_spec = tf.estimator.TrainSpec(\n    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'train.tfrecords'), 2),\n    max_steps=60000 * 2 / batch_size)\neval_spec = tf.estimator.EvalSpec(\n    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'test.tfrecords'), 1),\n    steps=10000 * 1 / batch_size,\n    start_delay_secs=0)\n\n# run !\ntf.estimator.train_and_evaluate(\n    mnist_fullyconnected_classifier,\n    train_spec,\n    eval_spec\n)\n\n# save model and variables\nmodel_dir = mnist_fullyconnected_classifier.export_savedmodel(\n    export_dir_base = FLAGS.model_folder,\n    serving_input_receiver_fn = _my_serving_input_fn)\nprint('current working directory is ', os.getcwd())\nprint('model is saved ', model_dir)")

#%% [markdown]
# ## Train on remote VM
# 
# Now let's start to integrate with AML services and run training on remote virtual machine.
#%% [markdown]
# ### Step 1 : Get workspace setting
# 
# Before starting, you must read your configuration settings. (See "[Exercise01 : Prepare Config Settings](/notebooks/exercise01_prepare_config.ipynb)")

#%%
from azureml.core import Workspace
import azureml.core

ws = Workspace.from_config()

#%% [markdown]
# ### Step 2 : Create new remote virtual machine
# 
# Create your new Data Science Virtual Machine (which is pre-configured for data science) with **GPU** (NC6). Before starting, please make sure to use NC6 supported location as workspace location. By enabling auto-scaling (from 0 to 1), you can save money (the node is terminated) if it's inactive.    
# If already exists, this script will get the existing one.
# 
# You can also attach an existing virtual machine (bring your own compute resource) as a compute target.

#%%
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

try:
    compute_target = ComputeTarget(workspace=ws, name='mydsvm01')
    print('found existing:', compute_target.name)
except ComputeTargetException:
    print('creating new.')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_NC6',
        min_nodes=0,
        max_nodes=1)
    compute_target = ComputeTarget.create(ws, 'mydsvm01', compute_config)
    compute_target.wait_for_completion(show_output=True)

#%% [markdown]
# ### Step 3 : Generate data reference config
# 
# You can configure to mount your preconfigured dataset (including train.tfrecords, test.tfrecords) from your ```Datastore``` in your compute target.    
# See "[Exercise02 : Prepare Datastore](/notebooks/exercise02_prepare_datastore.ipynb)".

#%%
from azureml.core import Datastore
from azureml.core.runconfig import DataReferenceConfiguration
# from azureml.data.data_reference import DataReference

# get your datastore (See "Exercise 02 : Prepare Datastore")
ds = Datastore.get(ws, datastore_name="myblob01")

# generate data reference configuration
dr_conf = DataReferenceConfiguration(
    datastore_name=ds.name,
    path_on_datastore='tfdata',
    mode='mount') # set 'download' if you copy all files instead of mounting

#%% [markdown]
# ### Step 4 : Generate config
# 
# Here we set docker environments for running scripts. We want to use ```Datastore``` as input data, so we set previous data reference configuration in this configuration.

#%%
from azureml.core.runconfig import RunConfiguration, DEFAULT_GPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies

run_config = RunConfiguration(
    framework="python",
    conda_dependencies=CondaDependencies.create(conda_packages=['tensorflow-gpu']))
run_config.target = compute_target.name
run_config.data_references = {ds.name: dr_conf}
run_config.environment.docker.enabled = True
run_config.environment.docker.gpu_support = True
run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE

#%% [markdown]
# ### Step 5 : Run script and wait for completion

#%%
from azureml.core import Experiment
from azureml.core import Run
from azureml.core import ScriptRunConfig

src = ScriptRunConfig(
    source_directory='./script',
    script='train.py',
    run_config=run_config,
    arguments=['--data_folder', str(ds.as_mount())]
)
# exp = Experiment(workspace=ws, name='test20181210-09')
exp = Experiment(workspace=ws, name='tf_remote_experiment')
run = exp.submit(config=src)
run.wait_for_completion(show_output=True)

#%% [markdown]
# ### Step 6 : Download results and check
#%% [markdown]
# Check generated files.

#%%
run.get_file_names()

#%% [markdown]
# Download model into your local machine.    
# **Please change ```1544491598``` to meet previous results.**

#%%
run.download_file(
    name='outputs/1544491598/saved_model.pb',
    output_file_path='remote_model/saved_model.pb')
run.download_file(
    name='outputs/1544491598/variables/variables.data-00000-of-00001',
    output_file_path='remote_model/variables/variables.data-00000-of-00001')
run.download_file(
    name='outputs/1544491598/variables/variables.index',
    output_file_path='remote_model/variables/variables.index')

#%% [markdown]
# Predict your test data using downloaded model.

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
pred_fn = tf.contrib.predictor.from_saved_model('./remote_model')
pred = pred_fn({'inputs': image_arr})

print('Predicted: ', pred['classes'].tolist())
print('Actual   : ', label_arr)

#%% [markdown]
# ### Step 7 : Remove AML compute
# 
# **You don't need to remove your AML compute** for saving money, because the nodes will be automatically terminated, when it's inactive.    
# But if you want to clean up, please run the following.

#%%
# Delete cluster (nbodes) and remove from AML workspace
mycompute = AmlCompute(workspace=ws, name='mydsvm01')
mycompute.delete()


#%%
# get a status for the current cluster.
print(mycompute.status.serialize())


#%%



