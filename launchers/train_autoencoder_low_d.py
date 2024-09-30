import hydra
import pathlib
import sys

this_file = pathlib.Path(__file__).absolute()
sys.path.append(str(this_file.parent.parent))


@hydra.main(
    version_base=None,
    config_path=str(this_file.parent.parent / 'configs'),
    config_name=this_file.stem
)
def main(config):
    from scripts import train_autoencoder_low_d
    train_autoencoder_low_d.main(config)


if __name__ == '__main__':
    main()
