# Use click (pip3 install click)

'''
Click is a Python package for creating beautiful command line interfaces in a composable way with as little code as necessary. 
It’s the “Command Line Interface Creation Kit”. It’s highly configurable but comes with sensible defaults out of the box.
'''

import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
@click.option('--run_as_dev', default=False, help='Run the class for dev purposes.', type=bool)
def hello(count, name, run_as_dev):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hello %s!' % name)

if __name__ == '__main__':
    hello()


# Usage from the command-line
python hello.py --count=3
python hello.py --help

