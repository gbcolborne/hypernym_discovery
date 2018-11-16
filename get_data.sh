#! /usr/bin/env bash

# Downloads data for SemEval 2018 task 9

print_usage() {
    echo -e "Usage: $0 [output-directory]"
    echo
    echo "Args:"
    echo -e "\t-output-directory"
}

if [ "$#" -ne 1 ]; then
    print_usage
    exit 1
fi
if [[ $1 == "-h" || $1 == "--help" ]]; then
    print_usage
    exit 0
fi

# Download data from Google Drive to output directory
dir_this=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $1
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qc812LUPMfcGTsH324O1gZzDuMLzFT_F' -O SemEval2018-Task9.zip
unzip SemEval2018-Task9.zip
rm SemEval2018-Task9.zip

# Copy task scorer in main directory, but do not include main
# function, which is not compatible with python 3, and which we don't
# need anyway.
mainline=$(awk "/if __name__ == '__main__':/ {print FNR}" SemEval2018-Task9/task9-scorer.py)
head -n $((mainline-1)) SemEval2018-Task9/task9-scorer.py > $dir_this/task9_scorer.py

