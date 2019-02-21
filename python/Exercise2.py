#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Exercise 02 : Prepare Datastore
# 
# Here we prepare ```Datastore``` for storing and sharing our dataset.
# 
# Before running the following script, **you must create your Storage Account** using [Azure Portal](https://portal.azure.com/), create container in blob, and retrieve your access key.
# 
# *back to [index](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/)*
#%% [markdown]
# ## Get config setting
# 
# Read your config settings. See "[Exercise01 : Prepare Config Settings](https://github.com/tsmatz/azure-ml-tensorflow-complete-sample/blob/master/notebooks/exercise01_prepare_config.ipynb)".

#%%
from azureml.core import Workspace
import azureml.core

ws = Workspace.from_config()

#%% [markdown]
# ## Use default datastore
# 
# The default datastore is attached in your AML workspace.    
# The data is stored in Azure File Share on *{your workspace name}{arbitary numbers}*.

#%%
# Get AML default datastore
ds = ws.get_default_datastore()

# Upload local "data" folder (incl. files) as "tfdata" folder
ds.upload(
    src_dir='./data',
    target_path='tfdata',
    overwrite=True)

#%% [markdown]
# ## Use your own blob storage
# 
# You can also use your own blob storage. Set your previously generated storage account name, key, and container.

#%%
from azureml.core import Datastore

ds = Datastore.register_azure_blob_container(
    ws,
    datastore_name='myblob01',
    account_name='amltest01',
    account_key='BAYcnjJ/TK...',
    container_name='container01',
    overwrite=True)

# Upload local "data" folder (incl. files) as "tfdata" folder
ds.upload(
    src_dir='./data',
    target_path='tfdata',
    overwrite=True)

#%% [markdown]
# Get the generated Datastore, and upload again.

#%%
# Get your own registered datastore
ds = Datastore.get(ws, datastore_name='myblob01')

# Upload local "data" folder (incl. files) as "tfdata" folder
ds.upload(
    src_dir='./data',
    target_path='tfdata',
    overwrite=True)


#%%

