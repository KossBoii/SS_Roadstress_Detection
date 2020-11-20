import json
import argparse
import re

def custom_default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", required=True)
    parser.add_argument("--dest", required=True)
    return parser

if __name__ == "__main__":
    args = custom_default_argument_parser().parse_args()
    print("Command Line Args:", args)

    with open(args.src) as f1:
        annos1 = json.load(f1)

    with open(args.dest) as f2:
        annos2 = json.load(f2)
    
    result_annos = {**annos2, **annos1}
    with open("result_annos.json", "w") as f:
        f.write(json.dumps(result_annos, separators=(',', ':')))
    f.close()
    
    str = ""
    with open("result_annos.json", "r") as f:
        str = f.read()
        re.sub('\n', '', str)
        re.sub(' ', '', str)

    with open("result_annos.json", "w") as f:
        f.write(str)