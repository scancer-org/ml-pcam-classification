# ML PCam Classification
[Scancer](https://scancer.org/) is a web application that uses
computer vision to assist with varying stages of breast cancer
detection. At its core is a collection of learnt detection models (in
[PyTorch](https://pytorch.org)) that are served as an API (using
[TorchServe](https://pytorch.org/serve/)).

This specific repository represents whole of the process of the ML modeling, including data preparation, of PCam Classification.

More info about the dataset can be found here - https://github.com/basveeling/pcam

Below you can find some information about how to train your own models using the infrastructre that we have created.
Specifically, it contains:

1. How to get the dataset
2. How to create a new model for PCAM-Classificiation
3. How to train a model
4. How to save a model (for integrating it using the "api" repo).


## 1. How to get the dataset
In order to get the dataset, one can download it from the PCAM repo (URL above), and store it in its Google Drive.

The dataset is ~7GB size, so it can definitely be downloaded "all at once".

Another method, that one can think of, is downloading the dataset (which is in HDF5 format), processing it, and saving in a different way. 
This method is not relevant for this repo at the moment, but it just an offer of another way of handling this.

Once downloaded, we stored the dataset in Google Drive, because the Colab notebooks are stored in Google as well. This makes the integration of dataset-code seamless easy.

## 2. How to create a new model for PCAM-Classification
Within this repo, one can find a folder called "notebooks". This folder contains several notebooks that were used as part of the infrastructure creation, model training, and whole "go-to" package of modeling the PCAM-Classification problem.

In order to create a new model, one can clone an existing notebook from the "notebooks" section (using the "Open in Colab" button, in each notebook).
The notebooks are ordered chronologically, and with an appropriate name per each, so one can look for a relevant notebook, and clone it from there.

In order to give one a sense, per May 8th, 2021, a relevant notebook can be `08_Fix_NaN_issue_in_division_cleaning_code_2021_05_01.ipynb`.

Once a notebook exists, it is important to make sure that the Drive integration actually works. This can be achieved by the below commands, for example:

`drive.mount('/content/gdrive/')`

`!ls gdrive/MyDrive/pcamv1`

First command mounts the drive into colab (creates a pointer so the data can be accessed easily).
Second command list the files in that particular folder (note: pcamv1 is the name we gave for the dataset, anyone else can have a different name).

The notebook contains a sketch of the whole process of training. Specifically, that is:
* Weights and Biases Installation (not a mandatory phase)
* Relevant Libraries Importing (where the model class, by the `cnn_model.py` script is imported)
* Weights and Biases parameters (not a mandatory phase)
* DataSet (PyTorch) definition
* Configurations (e.g. dataloader parameters, transformations, dataloader)
* Helper Functions (which are used in the process of training)
* Model Configuratoins (model instantiation, optimizer, learning rate, etc.)
* Training + Evaluation Process
* Model Saving

With regards to the model, as can be seen in the code, the notebook as a whole is a wrapper, and it expects one main instantiation:
`model = ModelCNN()`
ModelCNN is an instantiation of a script (`cnn_model.py`) which can be seen under `/src/models` 
After all, the model inherets a `nn.Module` model (from PyTorch), so theoretically, one can create its own model and train it.
Also, in case one is interested, it can use that same `cnn_model.py` script, and use it within its train process.
For that, the below model-specific-API is exposed:
`model = ModelCNN(n_input_channels, n_conv_output_channels, k, s, pad, p)`
n_input_channels - number of input channels that the model will get. Default value is 3.
n_conv_output_channels - number of convolutional filters that the model will get out after the input layer (aka: X).
k - kernel size. Default value is 3.
s - stride. Default value is 1.
pad - padding. Default value is 1.
p = Dropout rate (for regularization purposes). Default value is 0.5.

## 3. How to train a model
Once one has its model, it can start the training process.
Note: In case weights and biases are not used nor installed, it is recommended to remove/comment any wandb-relevant code.

## 4. How to save a model (for integrating it using the "api" repo)
Once the training has completed, you may want to save the model (for future purposes, e.g. hosting, batch prediction, real time prediction, etc.).
For that, one can use the following short script:
```
# generate a dummy input, in this case the input represents the PCAM image size (3x96x96)
example_input = torch.rand(1, 3, 96, 96).to(torch.device("cuda"))

# Store the existing model using torch.jit - make sure torch is imported!
traced_script_module = torch.jit.trace(model, example_input)

full_filename = "my_model_filename.pt"
# Save the script module under my_model_filename.pt
traced_script_module.save(full_filename)

# Test that loading works with no errors
new_model = torch.jit.load(full_filename)
```

Once the new trained model is saved, you may want to follow the "api" repository [documentation](https://github.com/scancer-org/api), in order to make predictions on a new dataset using it.

## Copyright and license

Copyright (c) 2021 [Harish Narayanan](https://harishnarayanan.org) and
[Daniel Hen](https://www.linkedin.com/in/daniel-hen/).

This code is licenced under the MIT Licence. See
[LICENSE](https://github.com/scancer-org/api/blob/main/LICENSE) for
the full text of this licence.

