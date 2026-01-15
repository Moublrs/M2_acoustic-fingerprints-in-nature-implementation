import argparse


from audio_to_spectrogram import navigate_directory_tree


def get_args_parser():
    parser = argparse.ArgumentParser('Spectrograms_from_dir', add_help=False)

    # Model parameters
    parser.add_argument('--input_directory', type=str,
                        help="""Name of the directory from which to collect the audio files.""")
    parser.add_argument('--output_directory', type=str,
                        help="""Name of the directory to which to drop the image files.""")
    parser.add_argument('--window_t', default=10, type=int,
                        help='Number of seconds of the temporal window in the spectrograms.')
    parser.add_argument("--min_f", default=0.0, type=float, help="""Minimum frequency of the spectrograms.""")
    parser.add_argument("--max_f", default=20000.0, type=float, help="""Maximum frequency of the spectrograms.""")
    parser.add_argument("--mel", action='store_true', help="""Generate MEL spectrograms.""")
    parser.add_argument("--rank", type=int, default=None,
                        help="Worker rank (for Windows parallelization)")
    parser.add_argument("--size", type=int, default=None,
                        help="Total number of workers (for Windows parallelization)")
    return parser


def main(args):
    size = args.size if args.size is not None else 1
    rank = args.rank if args.rank is not None else 0
    comm = None

    if args.window_t == None:
        args.window_t = 10

    if args.min_f == None:
        args.min_f = 0

    if args.max_f == None:
        args.max_f = 20000

    if args.mel == None:
        args.mel = False

    navigate_directory_tree(size=size, rank=rank, input_directory=args.input_directory,
                            output_directory=args.output_directory,
                            window_t=args.window_t,
                            min_f=args.min_f, max_f=args.max_f,
                           )

    if rank == 0:
        print('Task finished. Please review the output in case of any corrupted audio file.')
        print('The message for corrupted files is the following:')
        print(
            'XXXXXXXXXXXXX>>>>>>>>>>>>> SOMETHING SEEMS TO BE WRONG WITH THE INPUT FILE HERE:  /and/this/is/the/path/to/the/file.wav')
        print('Please, put the file in quarentine and re-run again sice such error kill the rank reading such a file.')
        print(
            'After you finish the task delete all txt files in the output directory tree since they were used for checkpointing')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Spectrograms_from_dir', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

