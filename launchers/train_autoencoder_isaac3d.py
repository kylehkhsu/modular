import hydra
import pathlib
import sys
import os

this_file = pathlib.Path(__file__).absolute()
sys.path.append(str(this_file.parent.parent))


@hydra.main(
    version_base=None,
    config_path=str(this_file.parent.parent / 'configs'),
    config_name=this_file.stem
)
def main(config):
    print(f'$HOME: {os.environ["HOME"]}')
    from scripts import train_autoencoder_isaac3d
    train_autoencoder_isaac3d.main(config)


if __name__ == '__main__':
    main()
