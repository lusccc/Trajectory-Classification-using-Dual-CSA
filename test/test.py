import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRJ_SEG_FEATURE')
    parser.add_argument('--feature_set_2',  type=str)
    args = parser.parse_args()
    print(args.feature_set_2)
    feature_set = [int(item) for item in args.feature_set_2.split(',')]
