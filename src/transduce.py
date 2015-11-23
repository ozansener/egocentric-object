from transduce_basic import BasicTransduction
import click


@click.command()
@click.option('--source', prompt='Source folder', help='Name of the source folder.')
@click.option('--target', prompt='Target Folder', help='Name of the target folder.')
def transduce(source, target):
    bt = BasicTransduction(source, target)
    bt.restore_the_model("s_MNIST_a_t_MNIST_r_model-19000")
    bt.featurize_source_and_target()


if __name__ == '__main__':
    transduce()