"""Soiled dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os


# followed
# https://stackoverflow.com/questions/69221972/how-to-load-custom-data-into-tfds-for-keras-cyclegan-example?noredirect=1&lq=1
# https://www.tensorflow.org/datasets/cli
# https://www.tensorflow.org/datasets/catalog/cycle_gan
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb#scrollTo=Ds4o1h4WHz9U


_DESCRIPTION = """
This Dataset contains Soiled and Unsoiled Images. 
A : Soiled
B : Unsoiled
Some of the  (Unsoiled) images are obtained from Rellis Dataset
This is under construction - alpha version 
"""

_CITATION = """\
  @article{
    author = {Krishna}
  }
"""
_DATA_OPTIONS = ["Soiled"]
_DL_URLS = {"Soiled":"C:\\Users\\kkosara\\Downloads\\Data_001.zip"}

class SoiledConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Soiled Class."""

  def __init__(self, *, data=None, **kwargs):
    """Constructs a DatasetaConfig.
    Args:
      data: `str`, one of `_DATA_OPTIONS`.
      **kwargs: keyword arguments forwarded to super.
    """
    if data not in _DATA_OPTIONS:
      raise ValueError("data must be one of %s" % _DATA_OPTIONS)

    super(SoiledConfig, self).__init__(**kwargs)
    self.data = data


class Soiled(tfds.core.GeneratorBasedBuilder):
  """Soiled dataset for CycleGAN."""

  BUILDER_CONFIGS = [
      SoiledConfig(  # pylint: disable=g-complex-comprehension
          name=config_name,
          version=tfds.core.Version("0.0.2"),
          release_notes={
              "0.0.1": "June 27th 2022. 1.2gb",
              "0.0.2": "Updated data on June 28th 2022. 6.8 gb  "
          },
          data=config_name,
      ) for config_name in _DATA_OPTIONS
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description="A dataset consisting of images from two classes A and "
        "B (For example: horses/zebras, apple/orange,...)",
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(names=["A", "B"]),
        }),
        supervised_keys=("image", "label"),
        homepage="https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    url = _DL_URLS[self.builder_config.name]
    data_dirs = dl_manager.download_and_extract(url)

    path_to_dataset = os.path.join(data_dirs, tf.io.gfile.listdir(data_dirs)[0])
    train_a_path = os.path.join(data_dirs, "trainA")
    train_b_path = os.path.join(data_dirs, "trainB")
    test_a_path = os.path.join(data_dirs, "testA")
    test_b_path = os.path.join(data_dirs, "testB")

    return [
        tfds.core.SplitGenerator(
            name="trainA", gen_kwargs={
                "path": train_a_path,
                "label": "A",
            }),
        tfds.core.SplitGenerator(
            name="trainB", gen_kwargs={
                "path": train_b_path,
                "label": "B",
            }),
        tfds.core.SplitGenerator(
            name="testA", gen_kwargs={
                "path": test_a_path,
                "label": "A",
            }),
        tfds.core.SplitGenerator(
            name="testB", gen_kwargs={
                "path": test_b_path,
                "label": "B",
            }),
    ]

  def _generate_examples(self, path, label):
    images = tf.io.gfile.listdir(path)

    for image in images:
      record = {
          "image": os.path.join(path, image),
          "label": label,
      }
      yield image, record
# Got the following error onlyway to fix it was CHANGING "path_to_dataset" in LINES 82,83,84 to "data_dirs"
# (tf_gpu) C:\Users\kkosara\Downloads\Soiled>tfds build 
# INFO[build.py]: Loading dataset  from path: C:\Users\kkosara\Downloads\Soiled\Soiled.py
# INFO[build.py]: download_and_prepare for dataset soiled/Soiled/0.0.2...
# INFO[dataset_builder.py]: Generating dataset soiled (C:\Users\kkosara\tensorflow_datasets\soiled\Soiled\0.0.2)
# Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to C:\Users\kkosara\tensorflow_datasets\soiled\Soiled\0.0.2...
# INFO[download_manager.py]: Skipping download of C:\Users\kkosara\Downloads\Data_001.zip: File cached in C:\Users\kkosara\tensorflow_datasets\downloads\Users_kkosara_Downloads_Data_18r38_Co4F-G6ka9wRk2wGFbDPqLZu8TekEV7s9L9enI.zip
# INFO[download_manager.py]: Reusing extraction of C:\Users\kkosara\tensorflow_datasets\downloads\Users_kkosara_Downloads_Data_18r38_Co4F-G6ka9wRk2wGFbDPqLZu8TekEV7s9L9enI.zip at C:\Users\kkosara\tensorflow_datasets\downloads\extracted\ZIP.Users_kkosara_Downloads_Data_18r38_Co4F-G6ka9wRk2wGFbDPqLZu8TekEV7s9L9enI.zip.
# Extraction completed...: 0 file [00:00, ? file/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 142.82 url/s] 
# Dl Size...: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7280528845/7280528845 [00:00<00:00, 909860891981.97 MiB/s] 
# Dl Completed...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 111.09 url/s] 
# wtf ['testA', 'testB', 'trainA', 'trainB']
# __data_dirs__ C:\Users\kkosara\tensorflow_datasets\downloads\extracted\ZIP.Users_kkosara_Downloads_Data_18r38_Co4F-G6ka9wRk2wGFbDPqLZu8TekEV7s9L9enI.zip
# ____path_to_dataset____ C:\Users\kkosara\tensorflow_datasets\downloads\extracted\ZIP.Users_kkosara_Downloads_Data_18r38_Co4F-G6ka9wRk2wGFbDPqLZu8TekEV7s9L9enI.zip\testA
# Traceback (most recent call last):
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\runpy.py", line 197, in _run_module_as_main
#     return _run_code(code, main_globals, None,
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\runpy.py", line 87, in _run_code
#     exec(code, run_globals)
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\Scripts\tfds.exe\__main__.py", line 7, in <module>
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\scripts\cli\main.py", line 102, in launch_cli
#     app.run(main, flags_parser=_parse_flags)
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\absl\app.py", line 312, in run
#     _run_main(main, args)
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\absl\app.py", line 258, in _run_main
#     sys.exit(main(argv))
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\scripts\cli\main.py", line 97, in main
#     args.subparser_fn(args)
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\scripts\cli\build.py", line 192, in _build_datasets
#     _download_and_prepare(args, builder)
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\scripts\cli\build.py", line 342, in _download_and_prepare
#     builder.download_and_prepare(
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\core\dataset_builder.py", line 481, in download_and_prepare
#     self._download_and_prepare(
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\core\dataset_builder.py", line 1218, in _download_and_prepare
#     future = split_builder.submit_split_generation(
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\core\split_builder.py", line 310, in submit_split_generation
#     return self._build_from_generator(**build_kwargs)
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow_datasets\core\split_builder.py", line 371, in _build_from_generator
#     for key, example in utils.tqdm(
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tqdm\std.py", line 1195, in __iter__
#     for obj in iterable:
#   File "C:\Users\kkosara\Downloads\Soiled\Soiled.py", line 114, in _generate_examples
#     images = tf.io.gfile.listdir(path)
#   File "C:\ProgramData\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow\python\lib\io\file_io.py", line 769, in list_directory_v2
#     raise errors.NotFoundError(
# tensorflow.python.framework.errors_impl.NotFoundError: Could not find directory C:\Users\kkosara\tensorflow_datasets\downloads\extracted\ZIP.Users_kkosara_Downloads_Data_18r38_Co4F-G6ka9wRk2wGFbDPqLZu8TekEV7s9L9enI.zip\testA\trainA
