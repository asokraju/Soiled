# Custom Tensorflow Input Pipeline for Cycle GANs

## Steps to create the dataset
1. Organize the data set inside a `Data.zip` file 
    ```
    trainA
    trainB
    testA
    testB
    ```
    `A` and `B` represents the two classes.

2. Provide the `path` ( of the `Data.zip` file ) in line `28` of `Soiled.py` i.e., 
    ```
    _DL_URLS = Soiled":"C:\\Users\\<user>\\Downloads\\Data_001.zip"}
    ```
3.  `cd` into `Soiled` folder and use `tfds build` command to build the data

4. The  Tensorflow record files can be found at `C:\Users\<user>\tensorflow_datasets\soiled`. If needed, these files can be taken elsewhere to use.
## Steps used to create the folder structure.

1. Install ``tensorflow_datasets`` package
2. On Command line type ``tfds new Soiled``. This will create a `Soiled` folder with file structure
    ``` __init__.py
    checksums.tsv
    dummy_data/
    Soiled.py
    Soiled_test.py
    ```
3. edit `Soiled.py` as needed.

## Possible issues:
1. If it fails to build the pipeline, delete the folder `tesorflow_datasets` folder BEFORE you retry. In windows it can found at `C\users\<user>`.
2. If it gives an error something similar to 
    ```
    # tensorflow.python.framework.errors_impl.NotFoundError: Could not find directory C:\Users\<user>\tensorflow_datasets\downloads\extracted\ZIP.Users_kkosara_Downloads_Data_18r38_Co4F-G6ka9wRk2wGFbDPqLZu8TekEV7s9L9enI.zip\testA\trainA
    ```
    try changing the `data_dirs` in lines to `path_to_dataset` or something that ensures it has the correct path to the downloaded data.
3. Ensure that the folder structure is proper 
    ```
        1. Organize the data set inside a `Data.zip` file 
        trainA
        trainB
        testA
        testB
        A and B represents the two classes.
    ```
    also ensure that there are nothing else except the image files inside the folder.

## Used Resources
1.  https://stackoverflow.com/questions/69221972/how-to-load-custom-data-into-tfds-for-keras-cyclegan-example?noredirect=1&lq=1
2. https://www.tensorflow.org/datasets/cli
3. https://www.tensorflow.org/datasets/catalog/cycle_gan
4. https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb#scrollTo=Ds4o1h4WHz9U


