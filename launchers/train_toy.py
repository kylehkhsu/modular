import hydra
import pathlib
import sys
this_file = pathlib.Path(__file__).absolute()
sys.path.append(str(this_file.parent.parent))

@hydra.main(version_base=None, config_path=str(this_file.parent.parent / 'configs'), config_name='train_toy')
def main(config):
    from scripts import train_toy
    train_toy.main(config)


if __name__ == '__main__':
    main()

