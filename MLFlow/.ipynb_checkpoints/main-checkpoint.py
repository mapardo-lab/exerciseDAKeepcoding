from utilsDataProc import read_raw_data, split_data, preprocess_data, read_proc_data
from utilsFeatures import extract_BoW_features, prepare_data
from utilsTrain import mlflow_knn_n
import argparse

def main(): 
    parser = argparse.ArgumentParser(description="Process one input file (--file1 OR --file2).")
    
    # create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file1", help="Path to first input file")
    group.add_argument("--file2", help="Path to second input file")

    args = parser.parse_args()

    # determine which file was provided
    file_path = args.file1 if args.file1 else args.file2
    
    if args.file1:
        # read raw data
        raw_data = read_raw_data(args.file1)
        print('Reading data ...')
        # split data into train/test
        reviewText_train, reviewText_test, overall_train, overall_test = split_data(raw_data)
        print('Splitting data ...')
        # data processing
        print('Processing data ...')
        # TODO Include write proc data file into preprocess_data
        words_train, words_test, labels_train, labels_test = preprocess_data(reviewText_train, reviewText_test, overall_train, overall_test)
    else:
        words_train, words_test, labels_train, labels_test = read_proc_data(args.file2)
    # features: bag of words
    print('Building features ...')
    features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test)
    # TODO Select features
    # prepare data for training
    print('Preparing features ...')
    X_train, X_test, y_train, y_test = prepare_data(features_train, features_test, labels_train, labels_test)
    # model train
    print('Training models ...')
    mlflow_knn_n('sentiment-Amazon2', X_train, y_train, range(1,21))
if __name__ == '__main__': 
    main()
