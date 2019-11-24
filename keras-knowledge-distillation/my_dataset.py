import tensorflow_datasets.public_api as tfds

class MyDataset(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""

  VERSION = tfds.core.Version('0.1.0')

def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=("This is the dataset for xxx. It contains yyy. The "
                     "images are kept at their original dimensions."),
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            "image_description": tfds.features.Text(),
            "image": tfds.features.Image(),
            # Here, labels can be of 5 distinct values.
            "label": tfds.features.ClassLabel(num_classes=5),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("image", "label"),
        # Homepage of the dataset for documentation
        homepage="https://dataset-homepage.org",
        # Bibtex citation for the dataset
        citation=r"""@article{my-awesome-dataset-2020,
                              author = {Smith, John},"}""",
    )

def _split_generators(self, dl_manager):
    # Download source data
    extracted_path = dl_manager.download_and_extract(...)

    # Specify the splits
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "images_dir_path": os.path.join(extracted_path, "train"),
                "labels": os.path.join(extracted_path, "train_labels.csv"),
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "images_dir_path": os.path.join(extracted_path, "test"),
                "labels": os.path.join(extracted_path, "test_labels.csv"),
            },
        ),
    ]

  def _generate_examples(self):
    # Yields examples from the dataset
    yield 'key', {}