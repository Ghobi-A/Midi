import argparse

from . import note_to_number, number_to_note


def main():
    parser = argparse.ArgumentParser(description="MIDI note converter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', '--note', help='Note name to convert to number')
    group.add_argument('-m', '--number', type=int, help='Number to convert to note')
    args = parser.parse_args()

    if args.note is not None:
        num = note_to_number(args.note)
        print(num)
    else:
        name = number_to_note(args.number)
        print(name)


if __name__ == '__main__':
    main()
