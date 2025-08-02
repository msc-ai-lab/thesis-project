MODELS_LIST = {
  'ViT16': 'google/vit-base-patch16-224',
  'ViT32': 'google/vit-base-patch32-384',
  'Anwarkh_ViT': 'Anwarkh1/Skin_Cancer-Image_Classification',
}

INPUT_SIZE_FOR_MODELS = {
  'Xception': (299, 299),
  'ViT16': (224, 224),
  'ViT32': (384, 384),
  'Anwarkh_ViT': (224, 224),
  'VGG16': (224, 224),
  'ResNet34': (384, 384),
  'ResNet50': (224, 224),
}