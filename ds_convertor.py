import os
from os import path
import argparse

DATA_TABLE = {
    
    0 : { 'p':1 , 'e':0 },
    1 : { 'b':1 , 'c':2 , 'x':3 , 'f':4 , 'k':5 , 's':6 },
    2 : { 'f':1 , 'g':2 , 'y':3 , 's':4 },
    3 : { 'n':1 , 'b':2 , 'c':3 , 'g':4 , 'r':5 , 'p':6 , 'u':7 , 'e':8 , 'w':9 , 'y':10 },
    4 : { 't':1 , 'f':0 },
    5 : { 'a':1 , 'l':2 , 'c':3 , 'y':4 , 'f':5 , 'm':6 , 'n':7 , 'p':8 , 's':9 },
    6 : { 'a':1 , 'd':2 , 'f':3 , 'n':4 },
    7 : { 'c':1 , 'w':2 , 'd':3 },
    8 : { 'b':1 , 'n':2 },
    9 : { 'k':1 , 'n':2 , 'b':3 , 'h':4 , 'g':5 , 'r':6 , 'o':7 , 'p':8 , 'u':9 , 'e':10 , 'w':11 , 'y':12 },
    10 : { 'e':1 , 't':2 },
    11 : { 'b':1 , 'c':2 , 'u':3 , 'e':4 , 'z':5 , 'r':6 , '?':7 },
    12 : { 'f':1 , 'y':2 , 'k':3 , 's':4 },
    13 : { 'f':1 , 'y':2 , 'k':3 , 's':4 },
    14 : { 'n':1 , 'b':2 , 'c':3 , 'g':4 , 'o':5 , 'p':6 , 'e':7 , 'w':8 , 'y':9 },
    15 : { 'n':1 , 'b':2 , 'c':3 , 'g':4 , 'o':5 , 'p':6 , 'e':7 , 'w':8 , 'y':9 },
    16 : { 'p':1 , 'u':2 },
    17 : { 'n':1 , 'o':2 , 'w':3 , 'y':4 },
    18 : { 'n':1 , 'o':2 , 't':3 },
    19 : { 'c':1 , 'e':2 , 'f':3 , 'l':4 , 'n':5 , 'p':6 , 's':7 , 'z':8 },
    20 : { 'k':1 , 'n':2 , 'b':3 , 'h':4 , 'r':5 , 'o':6 , 'u':7 , 'w':8 , 'y':9 },
    21 : { 'a':1 , 'c':2 , 'n':3 , 's':4 , 'v':5 , 'y':6 },
    22 : { 'g':1 , 'l':2 , 'm':3 , 'p':4 , 'u':5 , 'w':6 , 'd':7},
    
}

def convertData(data):
    data_indexed = data.split(',')
    new_data = list()
    for i , d in enumerate(data_indexed[:-1]):
        new_data.append(
            str(DATA_TABLE[i][d])
        )        

    return new_data

def convertDataSet(old_dataSet_path , new_dataSet_path):
    
    new_dataset = list()

    with open(old_dataSet_path , 'r') as odset:
        while(data := odset.readline()):
            converted_data = convertData(data)
            new_dataset.append(converted_data)

    with open(new_dataSet_path , 'w') as ndset:
        for data in new_dataset:
            writable_data = ','.join(data) + '\n'
            ndset.write(writable_data)


def main():
    parser = argparse.ArgumentParser(
        description='Convert the mushroom dataset to a number-based'
        'dataset that can be used for ML'
        )
    
    parser.add_argument('source', type=str, help='Source dataset location.')
    parser.add_argument( 'destination', type=str, default='new_mushroom_ds.ds',
        help='Converted files location (filename must be included).'
        )

    args = parser.parse_args()
    
    convertDataSet(args.source , args.destination)


if __name__ == '__main__':
    main()