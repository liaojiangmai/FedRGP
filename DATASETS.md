# Datasets Preparation

Place all datasets under the same folder (for example, `$DATA`) for easier
management.

The expected directory structure is:

```text
$DATA/
├── caltech-101/
├── dtd/
├── fgvc_aircraft/
├── food-101/
├── oxford_flowers/
├── oxford_pets/
└── ucf101/
```

If a dataset is already available elsewhere, create a symbolic link under
`$DATA` instead of downloading another copy.

> **Acknowledgement**: These dataset preparation instructions are adapted from
> the official [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

## OxfordPets

1. Create a folder named `oxford_pets/` under `$DATA`.
2. Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
3. Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
4. Download `split_zhou_OxfordPets.json` from [Google Drive](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing).

```text
oxford_pets/
├── images/
├── annotations/
└── split_zhou_OxfordPets.json
```

## Flowers102

1. Create a folder named `oxford_flowers/` under `$DATA`.
2. Download the images from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz.
3. Download the labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat.
4. Download `cat_to_name.json` from [Google Drive](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing).
5. Download `split_zhou_OxfordFlowers.json` from [Google Drive](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

```text
oxford_flowers/
├── cat_to_name.json
├── imagelabels.mat
├── jpg/
└── split_zhou_OxfordFlowers.json
```

## DTD

1. Download https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This creates `$DATA/dtd/`.
2. Download `split_zhou_DescribableTextures.json` from [Google Drive](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

```text
dtd/
├── images/
├── imdb/
├── labels/
└── split_zhou_DescribableTextures.json
```

## Caltech101

1. Create a folder named `caltech-101/` under `$DATA`.
2. Download http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract it under `$DATA/caltech-101/`.
3. Download `split_zhou_Caltech101.json` from [Google Drive](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and place it under `$DATA/caltech-101/`.

```text
caltech-101/
├── 101_ObjectCategories/
└── split_zhou_Caltech101.json
```

## Food101

1. Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract `food-101.tar.gz` under `$DATA`, resulting in `$DATA/food-101/`.
2. Download `split_zhou_Food101.json` from [Google Drive](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

```text
food-101/
├── images/
├── license_agreement.txt
├── meta/
├── README.txt
└── split_zhou_Food101.json
```

## FGVC Aircraft

1. Download https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
2. Extract `fgvc-aircraft-2013b.tar.gz` and keep only the `data/` folder.
3. Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

```text
fgvc_aircraft/
├── images/
└── ... # a collection of .txt files
```

## UCF101

1. Create a folder named `ucf101/` under `$DATA`.
2. Download `UCF-101-midframes.zip` from [Google Drive](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This archive contains extracted middle video frames.
3. Download `split_zhou_UCF101.json` from [Google Drive](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

```text
ucf101/
├── UCF-101-midframes/
└── split_zhou_UCF101.json
```
