from transduce_office import BasicTransduction
import click


@click.command()
@click.option('--source', prompt='Source folder', help='Name of the source folder.')
@click.option('--target', prompt='Target Folder', help='Name of the target folder.')
def transduce(source, target):
    bt = BasicTransduction(source, target)
    bt.restore_the_model("alexnet.npy")
    bt.featurize_source_and_target()
    for x in xrange(100):
       # one epoch
       bt.train_loop()

if __name__ == '__main__':
    transduce()
