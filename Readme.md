# Azure Machine Learning service Hands-On all for TensorFlow

This following Tutorial will take you through the deployment of Azure Machine Learning (AML) service using TensorFlow along with the entire Machine Learning development lifecycle of explore data, train, tune, and publish.

![](https://raw.githubusercontent.com/MicrosoftDocs/azure-docs/master/articles/machine-learning/service/media/overview-what-is-azure-ml/aml.png)

## Azure access for Students & Educators

 All student get $100 of Azure credit via Azure for Student for more details and get registered see [Azure Dev Tools for teaching] (https://azureforeducation.microsoft.com/en-US/Institutions)

## Data Resources

You can get [MNIST](http://yann.lecun.com/exdb/mnist/) dataset (**train.tfrecords**, **test.tfrecords**) in this example by running the following code, [https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py)

Microsoft Research Open Data Beta - Is a collection of free datasets from Microsoft Research to advance state-of-the-art research in areas such as natural language processing, computer vision, and domain specific sciences. Download or copy directly to a cloud-based Data Science Virtual Machine for a seamless development experience. see [https://msropendata.com/](https://msropendata.com/)

You can simply add any new data sets into your ```data``` folder.

## Exercises

- [Exercise01 : Prepare Config Settings](/notebooks/exercise01_prepare_config.ipynb)
- [Exercise02 : Prepare Datastore](/notebooks/exercise02_prepare_datastore.ipynb)
- [Exercise03 : Just Train in Your Working Machine](/notebooks/exercise03_train_simple.ipynb)
- [Exercise04 : Train on Remote GPU Virtual Machine](/notebooks/exercise04_train_remote.ipynb)
- [Exercise05 : Distributed Training](/notebooks/exercise05_train_distributed.ipynb)
- [Exercise06 : Experimentation Logs and Outputs](/notebooks/exercise06_experimentation.ipynb)
- [Exercise07 : Hyperparameter Tuning](/notebooks/exercise07_tune_hyperparameter.ipynb)
- [Exercise08 : Publish as a Web Service](/notebooks/exercise08_publish_model.ipynb)

Before starting, you must provision your environment as follows :

## 1. Setup your Virtual Machine and Conda Env

- Create Data Science Virtual Machine [DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) on Ubuntu (which also includes Azure ML CLI) using [Azure Portal](https://portal.azure.com/)

  Here we use DSVM, but you can also build your own environment from scratch.

  You will have to run some commands after the initial install to update your DSVM application and libararies

```
# Fetches the list of available updates
sudo apt-get update
# Strictly upgrades the current packages
sudo apt-get upgrade
# Installs updates (new ones)
sudo apt-get dist-upgrade
```

or you can do it all nicely with this single script

```
sudo bash -c 'for i in update {,dist-}upgrade auto{remove,clean}; do apt-get $i -y; done'
```

- Create conda virtual environment and activate as follows.

```
conda create -n myenv -y Python=3.6
# Update Conda Environment 
conda update -n base -c defaults conda 
conda activate myenv
```

- Install required packages in your conda environment (You must run in your conda env.)
 so please ensure you have used the command conda activate myenv

## Install Azure Machine Learning SDK

In the next step we will install  ```azureml-sdk[notebooks]``` installs notebook in your conda env and ```azureml_widgets``` extension (which is used in Exercise06) this Notebook extension is enabled in Jupyter. (See installed extension using ```jupyter nbextension list```.)
```
# install AML SDK
pip install azureml-sdk[notebooks]

# install notebook integration for conda
conda install nb_conda

# install required packages for development
# (use "tensorflow-gpu" if using GPU VM)
conda install -y matplotlib tensorflow
```

## 2. Create AML Workspace

Create new "Machine Learning services workspace" using [Azure Portal](https://portal.azure.com/) see [Creating Azure ML Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/studio/create-workspace)
Please make sure that **you must specify location (region) which supports NC-series (K80 GPU) virtual machines in workspace creation**, because workspace location is used when you create AML compute resources (virtual machines) in AML Python SDK. (See [here](https://azure.microsoft.com/en-us/global-infrastructure/services/?products=virtual-machines) for supported regions.)

## 3. Make Sure to Install ACI Provider in Your Azure Subscription

- Remove azure-ml-admin-cli extension on VM as follows. (This extension is already installed on DSVM and prevents you from running ```az login``` command.)

```
sudo -i az extension remove --name azure-ml-admin-cli
```

- Login to Azure using CLI

```
az login
```

- Check to see if ACI provider is already registered

```
az provider show -n Microsoft.ContainerInstance -o table
```

- If ACI is not registered, run the following command. (You should be the subscription owner to run this command.)

```
az provider register -n Microsoft.ContainerInstance
```

## 4. Start Jupyter Notebook

- Start jupyter notebook server in your conda environment.

```
jupyter notebook
```

- Copy url for notebook in the console output, and set SSH tunnel (port forwarding) on your desktop to access notebook.
  For instance, the following picture is the SSH tunnel setting on "putty" terminal client in Windows. (You can use ```ssh -L``` option in Mac OS.)
  ![SSH Tunnel settings with putty](/images/putty.png)

- Open your notebook url (http://localhost:8888/?token=...) using web browser in your desktop.
![Notebook Login](/images/Notebooks.png)

Simply paste into the password or token box the token recieved and press login this will load the Jupyter Hub

- Create new notebook by selecting "Python 3" kernel (which is your current conda environment).

Now you're ready to start !

References

Azure Machine Learning – [Notebooks & Resources](https://github.com/Azure/MachineLearningNotebooks)