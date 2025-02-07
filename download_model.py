import os
from pathlib import Path

import wandb
from wandb import Api
from omegaconf import OmegaConf

downloaded_conf_path = Path("conf/model_download/")
downloaded_tm_path = Path("tm/")

def download_models():
    """Download the models from the wandb run specified by the user."""

    # Update dependencies
    print("Please specify the URL of the wandb run in the format of 'team/competition/run_id'.\nFor example, 'team-epoch-iv/detect-harmful-brain-activity/s97lhfgc'.")
    print("To find this go to your wandb run -> Overview -> Run Path.")

    run_path = input("Please specify the run path of the wandb run: ").lower()

    api: Api = wandb.Api()
    run = api.run(f"{run_path}")

    is_sweep_run = input("Is this a sweep? (y/n): ").lower()

    # List all artefacts
    model_artefacts = []
    for artefact in run.logged_artifacts():
        if artefact.type == "model":
            print(artefact.name)
            model_artefacts.append(artefact)

    # Download all artefacts
    for artefact in model_artefacts:
        for file in artefact.files():
            print(f"Downloading {file.name} from {artefact.name}")
            file.download(root=downloaded_tm_path, exist_ok=True)

    # Get the config type artefact (should be only one)
    config_artefact = None
    for artefact in run.logged_artifacts():
        if artefact.type == "config":
            config_artefact = artefact
            break

    if not is_sweep_run:
        # Download the raw, human readable config file
        for file in config_artefact.files():
            if file.name not in ["config.yaml", "cv.yaml", "train.yaml"]:
                print(f"Downloading {file.name} from {config_artefact.name}")
                file.download(root=downloaded_conf_path, replace=True)
                # Rename the file to the name of the run
                os.rename(downloaded_conf_path / file.name, downloaded_conf_path / f"{run.name}.yaml")
                print(f"Renamed to {run.name}.yaml")

    else:
        # download the file called config.yaml from the artifact, which includes sweep parameters
        for file in config_artefact.files():
            if file.name == "config.yaml":
                print(f"Downloading {file.name} from {config_artefact.name}")
                file.download(root=downloaded_conf_path, replace=True)

                # read the file into a dictconfig with omegaconf
                config = OmegaConf.load(downloaded_conf_path / file.name)
                config = config.model
                # save the config again
                OmegaConf.save(config, downloaded_conf_path / file.name)

                # rename the file to the name of the run
                os.rename(downloaded_conf_path / file.name, downloaded_conf_path / f"{run.name}.yaml")
                print(f"Renamed to {run.name}.yaml")


if __name__ == "__main__":
    download_models()
