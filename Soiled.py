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
_DL_URLS = {"Soiled":"C:\\Users\\kkosara\\Downloads\\Data.zip"}

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
          version=tfds.core.Version("0.0.1"),
          release_notes={
              "0.0.1": "June 27th 2022. ",
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

    train_a_path = os.path.join(path_to_dataset, "trainA")
    train_b_path = os.path.join(path_to_dataset, "trainB")
    test_a_path = os.path.join(path_to_dataset, "testA")
    test_b_path = os.path.join(path_to_dataset, "testB")

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


# class Soiled(tfds.core.GeneratorBasedBuilder):
#   """DatasetBuilder for Soiled dataset."""

#   VERSION = tfds.core.Version('1.0.0')
#   RELEASE_NOTES = {
#       '1.0.0': 'Initial release.',
#   }

#   def _info(self) -> tfds.core.DatasetInfo:
#     """Returns the dataset metadata."""
#     # TODO(Soiled): Specifies the tfds.core.DatasetInfo object
#     return tfds.core.DatasetInfo(
#         builder=self,
#         description=_DESCRIPTION,
#         features=tfds.features.FeaturesDict({
#             # These are the features of your dataset like images, labels ...
#             'image': tfds.features.Image(shape=(None, None, 3)),
#             'label': tfds.features.ClassLabel(names=['no', 'yes']),
#         }),
#         # If there's a common (input, target) tuple from the
#         # features, specify them here. They'll be used if
#         # `as_supervised=True` in `builder.as_dataset`.
#         supervised_keys=('image', 'label'),  # Set to `None` to disable
#         homepage='https://dataset-homepage/',
#         citation=_CITATION,
#     )

#   def _split_generators(self, dl_manager: tfds.download.DownloadManager):
#     """Returns SplitGenerators."""
#     # TODO(Soiled): Downloads the data and defines the splits
#     path = dl_manager.download_and_extract('https://todo-data-url')

#     # TODO(Soiled): Returns the Dict[split names, Iterator[Key, Example]]
#     return {
#         'train': self._generate_examples(path / 'train_imgs'),
#     }

#   def _generate_examples(self, path):
#     """Yields examples."""
#     # TODO(Soiled): Yields (key, example) tuples from the dataset
#     for f in path.glob('*.jpeg'):
#       yield 'key', {
#           'image': f,
#           'label': 'yes',
#       }
