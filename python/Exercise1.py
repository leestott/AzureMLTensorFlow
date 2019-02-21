#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Exercise01 : Prepare Config Settings
# 
# Set your config in your current project folder.    
# (Before starting, you must create your Machine Learning service workspace with Azure Portal. See [Readme](/Readme.md).)
# 
# *back to [index](/Readme.md)*
#%% [markdown]
# ## Create config settings
# 
# You must create config setting into your project directory. The result was written in ```aml_config/config.json```.    
# If you're asked to login Azure with device login UI (https://microsoft.com/devicelogin), please open your browser and proceed to login.
# 
# This is needed **only once** in your project directory. edit the notebook cells below to enter your resource_group, workspace_name and subscription_id

#%%
from azureml.core import Workspace
ws = Workspace(
  workspace_name = "your AzureML WorkSpace",
  subscription_id = "your subscriptionID",
  resource_group = "your resource_group")
ws.write_config()

#%% [markdown]
# ## Check if you can see the generated config settings

#%%
my_workspace = Workspace.from_config()
my_workspace.get_details()


