#! /usr/bin/env bash

# Downloads corpus for SemEval 2018 task 9

print_usage() {
    echo -e "Usage: $0 [subtask output-directory]"
    echo
    echo "Args:"
    echo -e "\t-subtask one of {1A, 1B, 1C, 2A, 2B}"
    echo -e "\t-output-directory"
}

if [[ $1 == "-h" || $1 == "--help" ]]; then
    print_usage
    exit 0
fi
if [ "$#" -ne 2 ]; then
    print_usage
    exit 1
fi
if [[ ! $1 =~ 1A|1B|1C|2A|2B ]]; then
    print_usage
    exit 1
fi


# Download data from Google Drive to output directory
dir_this=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $2
if [[ $1 == "1A" ]]; then
    filename=1A_en_UMBC_tokenized.tar.gz
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz40_IukD5qDV0RsWUtDcDlYU2c' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz40_IukD5qDV0RsWUtDcDlYU2c" -O $filename && rm -rf /tmp/cookies.txt
    tar -xvzf $filename
    rm $filename
elif [[ $1 == "1B" ]]; then
    filename=1B_it_itwac_tokenized.tar.gz
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz40_IukD5qDdktiRnh3MjhyYWc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz40_IukD5qDdktiRnh3MjhyYWc" -O $filename && rm -rf /tmp/cookies.txt
    tar -xvzf $filename
    rm $filename
elif [[ $1 == "1C" ]]; then
    filename=1C_es_1Billion_tokenized.tar.gz
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz40_IukD5qDeUNOUFpuRjFzOGM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz40_IukD5qDeUNOUFpuRjFzOGM" -O $filename && rm -rf /tmp/cookies.txt
    tar -xvzf $filename
    rm $filename
elif [[ $1 == "2A" ]]; then
    filename=2A_med_pubmed_tokenized.tar.gz
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz40_IukD5qDekY2RXk0ekJRRGc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz40_IukD5qDekY2RXk0ekJRRGc" -O $filename && rm -rf /tmp/cookies.txt
    tar -xvzf $filename
    rm $filename
elif [[ $1 == "2B" ]]; then
    filename=2B_music_bioreviews_tokenized.tar.gz
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz40_IukD5qDUElXUnZ4YlJVYW8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz40_IukD5qDUElXUnZ4YlJVYW8" -O $filename && rm -rf /tmp/cookies.txt
    tar -xvzf $filename
    rm $filename
fi
