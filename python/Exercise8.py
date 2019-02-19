#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Exercise08 : Publish as a Web Service
# 
# Finally we publish our model as a web service with a few steps.
# 
# *back to [index](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/)*
#%% [markdown]
# ## Get workspace settings
# 
# Before starting, you must read your configuration settings. (See "[Exercise01 : Prepare Config Settings](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/blob/master/notebooks/exercise01_prepare_config.ipynb)".)

#%%
from azureml.core import Workspace
import azureml.core

ws = Workspace.from_config()

#%% [markdown]
# ## Train model
# 
# Let's train in your local environment and create model. (The model is saved in "```./outputs```" folder.)

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
import argparse
import math

import tensorflow as tf

FLAGS = None
batch_size = 100

#
# define functions for Estimator
#

def _my_input_fn(filepath, num_epochs):
    # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]
    # label - digit (0, 1, ..., 9)
    data_queue = tf.train.string_input_producer(
        [filepath],
        num_epochs = num_epochs) # data is repeated and it raises OutOfRange when data is over
    data_reader = tf.TFRecordReader()
    _, serialized_exam = data_reader.read(data_queue)
    data_exam = tf.parse_single_example(
        serialized_exam,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    data_image = tf.decode_raw(data_exam['image_raw'], tf.uint8)
    data_image.set_shape([784])
    data_image = tf.cast(data_image, tf.float32) * (1. / 255)
    data_label = tf.cast(data_exam['label'], tf.int32)
    data_batch_image, data_batch_label = tf.train.batch(
        [data_image, data_label],
        batch_size=batch_size)
    return {'inputs': data_batch_image}, data_batch_label

def _get_input_fn(filepath, num_epochs):
    return lambda: _my_input_fn(filepath, num_epochs)

def _my_model_fn(features, labels, mode):
    # with tf.device(...): # You can set device if using GPUs

    # define network and inference
    # (simple 2 fully connected hidden layer : 784->128->64->10)
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal(
                [784, FLAGS.first_layer],
                stddev=1.0 / math.sqrt(float(784))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([FLAGS.first_layer]),
            name='biases')
        hidden1 = tf.nn.relu(tf.matmul(features['inputs'], weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal(
                [FLAGS.first_layer, FLAGS.second_layer],
                stddev=1.0 / math.sqrt(float(FLAGS.first_layer))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([FLAGS.second_layer]),
            name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal(
                [FLAGS.second_layer, 10],
                stddev=1.0 / math.sqrt(float(FLAGS.second_layer))),
        name='weights')
        biases = tf.Variable(
            tf.zeros([10]),
            name='biases')
        logits = tf.matmul(hidden2, weights) + biases
 
    # compute evaluation matrix
    predicted_indices = tf.argmax(input=logits, axis=1)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label_indices = tf.cast(labels, tf.int32)
        accuracy = tf.metrics.accuracy(label_indices, predicted_indices)
        tf.summary.scalar('accuracy', accuracy[1]) # output to TensorBoard
 
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits)
 
    # define operations
    if mode == tf.estimator.ModeKeys.TRAIN:
        #global_step = tf.train.create_global_step()
        #global_step = tf.contrib.framework.get_or_create_global_step()
        global_step = tf.train.get_or_create_global_step()        
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': accuracy
        }
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs)

def _my_serving_input_fn():
    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

#
# Main
#

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_folder',
    type=str,
    default='./data',
    help='Folder path for input data')
parser.add_argument(
    '--chkpoint_folder',
    type=str,
    default='./logs',  # AML experiments logs folder
    help='Folder path for checkpoint files')
parser.add_argument(
    '--model_folder',
    type=str,
    default='./outputs',  # AML experiments outputs folder
    help='Folder path for model output')
parser.add_argument(
    '--learning_rate',
    type=float,
    default='0.07',
    help='Learning Rate')
parser.add_argument(
    '--first_layer',
    type=int,
    default='128',
    help='Neuron number for the first hidden layer')
parser.add_argument(
    '--second_layer',
    type=int,
    default='64',
    help='Neuron number for the second hidden layer')
FLAGS, unparsed = parser.parse_known_args()

# clean checkpoint and model folder if exists
if os.path.exists(FLAGS.chkpoint_folder) :
    for file_name in os.listdir(FLAGS.chkpoint_folder):
        file_path = os.path.join(FLAGS.chkpoint_folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
if os.path.exists(FLAGS.model_folder) :
    for file_name in os.listdir(FLAGS.model_folder):
        file_path = os.path.join(FLAGS.model_folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# read TF_CONFIG
run_config = tf.contrib.learn.RunConfig()

# create Estimator
mnist_fullyconnected_classifier = tf.estimator.Estimator(
    model_fn=_my_model_fn,
    model_dir=FLAGS.chkpoint_folder,
    config=run_config)
train_spec = tf.estimator.TrainSpec(
    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'train.tfrecords'), 2),
    max_steps=60000 * 2 / batch_size)
eval_spec = tf.estimator.EvalSpec(
    input_fn=_get_input_fn(os.path.join(FLAGS.data_folder, 'test.tfrecords'), 1),
    steps=10000 * 1 / batch_size,
    start_delay_secs=0)

# run !
tf.estimator.train_and_evaluate(
    mnist_fullyconnected_classifier,
    train_spec,
    eval_spec
)

# save model and variables
model_dir = mnist_fullyconnected_classifier.export_savedmodel(
    export_dir_base = FLAGS.model_folder,
    serving_input_receiver_fn = _my_serving_input_fn)
print('current working directory is ', os.getcwd())
print('model is saved ', model_dir)

#%% [markdown]
# ## Archive your model as model.zip

#%%
import shutil
shutil.make_archive('model', 'zip', root_dir=model_dir)

#%% [markdown]
# ## Register model into model management

#%%
from azureml.core.model import Model

registered_model = Model.register(
    model_path = './model.zip',
    model_name = 'sample_model',
    workspace = ws)

#%% [markdown]
# ## Deploy as web service
#%% [markdown]
# First you generate your scoring source. (See "[Exercise03 : Just Train in Your Working Machine](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/blob/master/notebooks/exercise03_train_simple.ipynb)" for the original source code.)    
# This should include ```init()``` and ```run()```.

#%%
get_ipython().run_cell_magic(u'writefile', u'score.py', u"import json\nimport zipfile\nimport tensorflow as tf\nfrom azureml.core.model import Model\n\ndef init():\n    global pred_fn\n    model_path = Model.get_model_path(model_name='sample_model')\n    with zipfile.ZipFile(model_path) as target_zip:\n        target_zip.extractall('extracted_model')\n    pred_fn = tf.contrib.predictor.from_saved_model('./extracted_model')\n\ndef run(raw_data):\n    try:\n       data = json.loads(raw_data)['data']\n       result = pred_fn({'inputs': data})\n       return result['classes'].tolist()\n    except Exception as e:\n       result = str(e)\n       return 'Internal Exception : ' + result")

#%% [markdown]
# Create deploy configuration for preparation.

#%%
from azureml.core.webservice import AciWebservice, Webservice

aci_conf = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1, 
    description='This is a tensorflow example.')

#%% [markdown]
# Create image configuration for preparation.

#%%
from azureml.core.image import ContainerImage
from azureml.core.conda_dependencies import CondaDependencies 

# Generate conda dependency file
conda_dependency = CondaDependencies.create()
conda_dependency.add_pip_package('tensorflow')
### Or you can also write as follows (make sure to insert 'azureml-defaults' module)
#conda_dependency = CondaDependencies.create(pip_packages=['azureml-defaults', 'tensorflow'])
with open('myenv.yml', 'w') as f:
    f.write(conda_dependency.serialize_to_string())

# Create image configuration
image_conf = ContainerImage.image_configuration(
    execution_script="score.py",
    runtime="python",
    conda_file="myenv.yml")

#%% [markdown]
# Deploy as a web service !

#%%
svc = Webservice.deploy_from_model(
    name='my-mnist-service',
    deployment_config=aci_conf,
    models=[registered_model],
    image_config=image_conf,
    workspace=ws)
svc.wait_for_deployment(show_output=True)


#%%
# See details, if error has occured
print(svc.get_logs())

#%% [markdown]
# Check service url

#%%
svc.scoring_uri

#%% [markdown]
# ## Test your web service

#%%
import requests
import json

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
        image_arr.append(image.tolist())
        label_arr.append(label)

# Invoke web service !
headers = {'Content-Type':'application/json'}
# for AKS deployment you'd need to the service key in the header as well
# api_key = svc.get_key()
# headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)} 
values = json.dumps(image_arr)
input_data = "{\"data\": " + values + "}"
http_res = requests.post(
    svc.scoring_uri,
    input_data,
    headers = headers)
print('Predicted : ', http_res.text)
print('Actual    : ', label_arr)

#%% [markdown]
# ## Remove service

#%%
svc.delete()


#%%



