import argparse
import sys
from stereosegger.cli.create_dataset_fast import main as create_dataset_main
from stereosegger.cli.train_model import main as train_main
from stereosegger.cli.predict import main as predict_main
from stereosegger.cli.predict_fast import main as predict_fast_main
from stereosegger.cli.convert_saw_h5ad_to_segger_parquet import main as convert_saw_main


def main():
    parser = argparse.ArgumentParser(
        description="Command line interface for the StereoSegger segmentation package",
        usage="""stereosegger <command> [<args>]

Available commands:
   convert_saw       Convert Stereo-seq SAW H5AD to Parquet
   create_dataset    Create Segger dataset from spatial transcriptomics data
   train_model       Train the Segger segmentation model
   predict           Run the Segger segmentation model
   predict_fast      Run the Segger segmentation model (fast version)
""",
    )
    parser.add_argument("command", help="Subcommand to run")

    # parse_args defaults to [1:] for args, but you can specify your own to parse a subset of args
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(sys.argv[1:2])
    
    # save old argv
    old_argv = sys.argv[:]
    
    if args.command == "convert_saw":
        sys.argv = [old_argv[0] + " convert_saw"] + old_argv[2:]
        convert_saw_main()
    elif args.command == "create_dataset":
        sys.argv = [old_argv[0] + " create_dataset"] + old_argv[2:]
        create_dataset_main()
    elif args.command == "train_model":
        sys.argv = [old_argv[0] + " train_model"] + old_argv[2:]
        train_main()
    elif args.command == "predict":
        sys.argv = [old_argv[0] + " predict"] + old_argv[2:]
        predict_main()
    elif args.command == "predict_fast":
        sys.argv = [old_argv[0] + " predict_fast"] + old_argv[2:]
        predict_fast_main()
    else:
        print(f"Unrecognized command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()