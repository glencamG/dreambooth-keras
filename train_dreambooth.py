import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import math

import tensorflow as tf
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler

# from ddim import DDIMScheduler as NoiseScheduler

import tensorflow as tf
from tensorflow.keras import mixed_precision

from src import utils
from src.constants import MAX_PROMPT_LENGTH
from src.datasets import DatasetUtils
from src.dreambooth_trainer import DreamBoothTrainer
from src.utils import QualitativeValidationCallback, DreamBoothCheckpointCallback

import wandb
from wandb.keras import WandbMetricsLogger


# These hyperparameters come from this tutorial by Hugging Face:
# https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
def get_optimizer(
    lr=5e-6, beta_1=0.9, beta_2=0.999, weight_decay=(1e-2,), epsilon=1e-08
):
    """Instantiates the AdamW optimizer."""

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )

    return optimizer


def prepare_trainer(
    img_resolution: int,
    train_text_encoder: bool,
    use_mp: bool,
    pretrained_url_kwargs: dict,
    optimizer_kwargs: dict,
    **kwargs,
):
    vae_weights_url = pretrained_url_kwargs["vae"]
    model_weights_url = pretrained_url_kwargs["model"]
    is_download_vae = vae_weights_url == None
    is_download_model = model_weights_url == None

    """Instantiates and compiles `DreamBoothTrainer` for training."""
    image_encoder = ImageEncoder(
        img_resolution, img_resolution, download_weights=is_download_vae
    )

    dreambooth_trainer = DreamBoothTrainer(
        diffusion_model=DiffusionModel(
            img_resolution,
            img_resolution,
            MAX_PROMPT_LENGTH,
            download_weights=is_download_model,
        ),
        # Remove the top layer from the encoder, which cuts off
        # the variance and only returns the mean.
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        ),
        noise_scheduler=NoiseScheduler(),
        train_text_encoder=train_text_encoder,
        use_mixed_precision=use_mp,
        **kwargs,
    )

    if vae_weights_url:
        vae_weights_fpath = tf.keras.utils.get_file(origin=vae_weights_url)
        image_encoder.load_weights(vae_weights_fpath)
    if model_weights_url:
        model_weights_fpath = tf.keras.utils.get_file(origin=model_weights_url)
        dreambooth_trainer.diffusion_model.load_weights(model_weights_fpath)

    optimizer = get_optimizer(**optimizer_kwargs)
    dreambooth_trainer.compile(optimizer=optimizer, loss="mse")
    print("DreamBooth trainer initialized and compiled.")

    return dreambooth_trainer


def train(dreambooth_trainer, train_dataset, max_train_steps, callbacks):
    """Performs DreamBooth training `DreamBoothTrainer` with the given `train_dataset`."""
    num_update_steps_per_epoch = train_dataset.cardinality()
    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    print(f"Training for {epochs} epochs.")

    dreambooth_trainer.fit(train_dataset, epochs=epochs, callbacks=callbacks)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to perform DreamBooth training using Stable Diffusion."
    )
    # Dataset related.
    parser.add_argument(
        "--instance_images_url",
        default="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz",
        type=str,
    )
    parser.add_argument(
        "--class_images_url",
        default="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz",
        type=str,
    )
    parser.add_argument("--unique_id", default="sks", type=str)
    parser.add_argument("--class_category", default="dog", type=str)
    parser.add_argument("--img_resolution", default=512, type=int)
    # Model weights
    parser.add_argument("--pretrained_model_url", default=None, type=str)
    parser.add_argument("--pretrained_vae_url", default=None, type=str)
    # Optimization hyperparameters.
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    # Training hyperparameters.
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_train_steps", default=800, type=int)
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="If fine-tune the text-encoder too.",
    )
    parser.add_argument(
        "--mp", action="store_true", help="Whether to use mixed-precision."
    )
    # Misc.
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for experiment tracking.",
    )
    parser.add_argument(
        "--validation_prompts",
        nargs="+",
        default=None,
        type=str,
        help="Prompts to generate samples for validation purposes and logging on Weights & Biases",
    )
    parser.add_argument(
        "--num_images_to_generate",
        default=5,
        type=int,
        help="Number of validation image to generate per prompt.",
    )

    return parser.parse_args()


def run(args):
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(args.seed)

    validation_prompts = [
        f"A photo of {args.unique_id} {args.class_category} in a bucket"
    ]
    if args.validation_prompts is not None:
        validation_prompts = args.validation_prompts

    run_name = f"lr@{args.lr}-max_train_steps@{args.max_train_steps}-train_text_encoder@{args.train_text_encoder}"
    if args.log_wandb:
        wandb.init(project="dreambooth-keras", name=run_name, config=vars(args))

    if args.mp:
        print("Enabling mixed-precision...")
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"

    print("Initializing dataset...")
    data_util = DatasetUtils(
        instance_images_url=args.instance_images_url,
        class_images_url=args.class_images_url,
        unique_id=args.unique_id,
        class_category=args.class_category,
        train_text_encoder=args.train_text_encoder,
        batch_size=args.batch_size,
    )
    train_dataset = data_util.prepare_datasets()

    print("Initializing trainer...")
    ckpt_path_prefix = run_name
    optimizer_kwargs = {
        "lr": args.lr,
        "beta_1": args.beta_1,
        "beta_2": args.beta_2,
        "epsilon": args.epsilon,
        "weight_decay": args.wd,
    }
    pretrained_url_kwargs = {
        "vae": args.pretrained_vae_url,
        "model": args.pretrained_model_url,
    }
    dreambooth_trainer = prepare_trainer(
        args.img_resolution,
        args.train_text_encoder,
        args.mp,
        pretrained_url_kwargs,
        optimizer_kwargs,
    )

    callbacks = [
        # save model checkpoint and optionally log model checkpoints to
        # Weights & Biases as artifacts
        DreamBoothCheckpointCallback(ckpt_path_prefix, save_weights_only=True)
    ]
    if args.log_wandb:
        # log training metrics to Weights & Biases
        callbacks.append(WandbMetricsLogger(log_freq="batch"))
        # perform inference on validation prompts at the end of every epoch and
        # log the resuts to a Weights & Biases table
        callbacks.append(
            QualitativeValidationCallback(
                img_heigth=args.img_resolution,
                img_width=args.img_resolution,
                prompts=validation_prompts,
                num_imgs_to_gen=args.num_images_to_generate,
            )
        )

    train(dreambooth_trainer, train_dataset, args.max_train_steps, callbacks)

    if args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    run(args)
