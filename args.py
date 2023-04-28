"""
Module containing the classes for arguments in a CLI for a equity backtesting program.
"""
import argparse


class ArgumentParser:
    """
    ArgumentParser is an argparse class that takes command line input (see TEST_SCRIPT.cmd) for usage
    """

    @classmethod
    def set_parser(cls):
        """
        An Args object with information parsed from the command line inputs.

        :return: An Args object.
        """
        parser = argparse.ArgumentParser()

        parser_mutually_exclusive_group_input = parser.add_mutually_exclusive_group(required=True)

        parser_mutually_exclusive_group_input.add_argument('--inputfile', type=str, dest='input_file',
                                                           help="Path of the input file.")

        parser_mutually_exclusive_group_input.add_argument('--inputdata', type=str, dest='input_data',
                                                           help="Request data inputted through command line if an "
                                                                "input file is not used.")

        parser.add_argument('--expanded', action='store_true', help='(Optional) Shows additional information'
                                                                    ' of certain attributes ')

        parser.add_argument('--output', type=str, dest='output_file', help='Path of the output')

        parser.add_argument('mode', type=str,
                            choices=["pokemon", "ability", "move"],
                            help="The mode of the program. Program can provide information about a pokemon for these "
                                 "options: 'pokemon', 'ability', or 'move')")

        kwarg = vars(parser.parse_args())
        return Args(**kwarg)


class Args:
    """
    Arguments has the values needed to make a request to get pokemon data.
    """

    def __init__(self, mode: str, input_data: str, expanded: bool,
                 input_file: str = None, output_file: str = None):
        """
        Constructor.

        :param mode: a string, the input mode.
        :param input_data str: input data instead of input file
        :param expanded: bool, extra information
        :param input_file: str, path of input file
        :param output_file: str, data of the output file
        """
        self.mode = mode
        self.input_data = input_data
        self.expanded = expanded
        self.input_file = input_file
        self.output_file = output_file

    def __str__(self):
        """
        To string method, returns the string representation of this object.
        :return: a String.
        """
        return f'Arguments: {str(vars(self))}'
