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

## loading the data
There are multiple ways to do it. 
1. Import the necessary packages:
    ```
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import sys
    ```
2. Ensure that the path to `Soiled` folder containg the code NOT the data generated. For this I have added the path as follows:
    ```
    sys.path.insert(1, 'C:\\Users\\<user>\\Downloads\\')
    ```
3. Then the data can be loaded using:
    ```
    ds = tfds.load('Soiled')
    ds
    ```
    ```
    {'trainA': <PrefetchDataset shapes: {image: (None, None, 3), label: ()}, types: {image: tf.uint8, label: tf.int64}>,
    'trainB': <PrefetchDataset shapes: {image: (None, None, 3), label: ()}, types: {image: tf.uint8, label: tf.int64}>,
    'testA': <PrefetchDataset shapes: {image: (None, None, 3), label: ()}, types: {image: tf.uint8, label: tf.int64}>,
    'testB': <PrefetchDataset shapes: {image: (None, None, 3), label: ()}, types: {image: tf.uint8, label: tf.int64}>}
    ```
4. test:
    ```
    next(iter(ds['trainA']))
    ```
    ```
    Output exceeds the size limit. Open the full output data in a text editor
    {'image': <tf.Tensor: shape=(1200, 1920, 3), dtype=uint8, numpy=
    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [115, 173, 187],
            [112, 174, 197],
            [108, 172, 199]],
    
            [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [119, 170, 191],
            [115, 165, 192],
            [117, 168, 197]],
    
            [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [109, 145, 179],
            [134, 162, 199],
            [134, 158, 194]],
    
    ...
            ...,
            [ 72,  95,  67],
            [ 78,  99,  66],
            [ 79,  99,  62]]], dtype=uint8)>,
    'label': <tf.Tensor: shape=(), dtype=int64, numpy=0>}
    ```

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


