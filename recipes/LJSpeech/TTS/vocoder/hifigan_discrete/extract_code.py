"""
Apply K-means clustering over acoustic features to extract speech units for HiFi-GAN training.

Authors
 * Jarod Duret 2023
"""

import json
import logging
import pathlib as pl

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.lobes.models.huggingface_transformers import (
    hubert,
    wav2vec2,
    wavlm,
)
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import (
    DiscreteSSL,
)

OPT_FILE = "opt_ljspeech_extract_code.pkl"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"

ENCODER_CLASSES = {
    "HuBERT": hubert.HuBERT,
    "Wav2Vec2": wav2vec2.Wav2Vec2,
    "WavLM": wavlm.WavLM,
}


def setup_logger():
    """Set up a logger with a log format and logging level."""
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_device(use_cuda):
    """Determine and return the appropriate device for computation."""
    use_cuda = use_cuda and torch.cuda.is_available()
    print("\n" + "=" * 30)
    print("USE_CUDA SET TO: {}".format(use_cuda))
    print("CUDA AVAILABLE?: {}".format(torch.cuda.is_available()))
    print("=" * 30 + "\n")
    return torch.device("cuda" if use_cuda else "cpu")


def np_array(tensor):
    """Convert a Pytorch tensor to a Numpy array."""
    tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()
    return tensor.numpy()


def skip(splits, save_folder, conf):
    """
    Detects if the ljspeech data_extraction has been already done.
    If the extraction has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    split_files = {
        "train": TRAIN_JSON,
        "valid": VALID_JSON,
        "test": TEST_JSON,
    }

    for split in splits:
        if not (save_folder / split_files[split]).exists():
            skip = False

    #  Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(save_opt.as_posix())
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False
    return skip


def extract_ljspeech(
    data_folder,
    splits,
    kmeans_folder,
    kmeans_dataset,
    num_clusters,
    encoder_type,
    encoder_source,
    layer,
    encoder_save_folder,
    codes_save_folder,
    sample_rate=16000,
    skip_extract=False,
):
    """
    Extract speech units for HiFi-GAN training on the LJspeech datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LJspeech dataset is stored.
    splits : list
        List of splits to prepare.
    kmeans_folder: str
        Huggingface repository if that contains the pretrained kmean model.
    kmeans_dataset : str
        Name of the dataset that Kmeans model on HF repo is trained with.
    num_clusters:  (int)
        determine the number of clusters of the targeted kmeans models to be downloaded.
    encoder_type: str
        Name of the model used as feature extractor.
    encoder_source: str
        Url to the model used as feature extractor.
    layer: List[int] (default: [7]):
        Determine which layers of SSL should be used to extract information.
    encoder_save_folder: str
        Path to the folder where the ssl encoder stored.
    codes_save_folder: str
        Path to the folder where the tokens are stored.
    sample_rate: int
        LjSpeech dataset sample rate
    skip_extract: Bool
        If True, skip extraction.

    Example
    -------
    >>> from recipes.LJSpeech.TTS.vocoder.hifi_gan_unit.extract_code import extract_ljspeech
    >>> data_folder = 'data/LJspeech/'
    >>> splits = ['train', 'valid']
    >>> kmeans_folder = 'speechbrain/SSL_Quantization'
    >>> kmeans_dataset = LibriSpeech-100-360-500
    >>> encoder_type = 'HuBERT'
    >>> encoder_source = facebook/hubert-large-ll60k
    >>> layer = [7]
    >>> encoder_save_folder = 'ssl_encoder/'
    >>> codes_save_folder = 'codes/'
    >>> extract_ljspeech(data_folder, splits, kmeans_folder, kmeans_filename, encoder_type, encoder_source, layer, encoder_save_folder, codes_save_folder)
    """
    logger = setup_logger()

    if skip_extract:
        return
    # Create configuration for easily skipping code extraction stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "save_folder": codes_save_folder,
        "kmeans_folder": kmeans_folder,
        "encoder_type": encoder_type,
        "encoder_source": encoder_source,
        "layer": layer,
    }

    codes_save_folder = pl.Path(codes_save_folder)
    # Check if this phase is already done (if so, skip it)
    if skip(splits, codes_save_folder, conf):
        logger.info("Skipping code extraction, completed in previous run.")
        return

    # Fetch device
    device = get_device(use_cuda=True)

    save_opt = codes_save_folder / OPT_FILE
    data_folder = pl.Path(data_folder)
    encoder_save_folder = pl.Path(encoder_save_folder)
    codes_save_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading encoder: {encoder_source} ...")
    if encoder_type not in ENCODER_CLASSES:
        raise TypeError("Not a supported Encoder")

    encoder_class = ENCODER_CLASSES[encoder_type]
    encoder = encoder_class(
        source=encoder_source,
        save_path=encoder_save_folder.as_posix(),
        output_norm=False,
        freeze=True,
        freeze_feature_extractor=True,
        apply_spec_augment=False,
        output_all_hiddens=True,
    ).to(device)

    discrete_encoder = DiscreteSSL(
        save_path=encoder_save_folder.as_posix(),
        ssl_model=encoder,
        kmeans_dataset=kmeans_dataset,
        kmeans_repo_id=kmeans_folder,
        num_clusters=num_clusters,
    )

    for split in splits:
        dataset_path = data_folder / f"{split}.json"
        logger.info(f"Reading dataset from {dataset_path} ...")
        meta_json = json.load(open(dataset_path))
        for key in tqdm(meta_json.keys()):
            item = meta_json[key]
            wav = item["wav"]
            with torch.no_grad():
                info = torchaudio.info(wav)
                audio = sb.dataio.dataio.read_audio(wav)
                audio = torchaudio.transforms.Resample(
                    info.sample_rate,
                    sample_rate,
                )(audio)
                audio = audio.unsqueeze(0).to(device)
                deduplicates = [False for _ in layer]
                bpe_tokenizers = [None for _ in layer]
                tokens, _, _ = discrete_encoder(
                    audio,
                    SSL_layers=layer,
                    deduplicates=deduplicates,
                    bpe_tokenizers=bpe_tokenizers,
                )
                tokens = np_array(tokens.squeeze(0))
            np.save(codes_save_folder / f"{key}.npy", tokens)

    logger.info("Extraction completed.")
    save_pkl(conf, save_opt)
