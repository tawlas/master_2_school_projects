import sys
import os
from pyspark import SparkContext
import time

# Functions ---------------------------------------
def map_function(line):
    persons = [p.strip() for p in line.split(",")]
    x = persons[0]
    F = []
    for i in range(1, len(persons)):
        for j in range(i+1, len(persons)):
            if persons[i] < persons[j]:
                F.append(((i, j), x))
    return F

def get_common_friends(text_file):
    mapped_friends = text_file.flatMap(map_function)
    common_unique = mapped_friends.distinct()
    common_friends = common_unique.groupByKey().mapValues(list)
    return common_friends

# Main code -------------------------------------------
input_path = "hdfs://sar01:9000/data/sn/"
output_path = "hdfs://sar01:9000/cpupsmia1/cpupsmia1_12/"
file_name = os.path.basename(sys.argv[1])
input_file_name = input_path + file_name
output_file_name = output_path + os.path.splitext(file_name)[0]+".out"

# - create the Spark context and open the input file as a RDD (in the HDFS)
sc = SparkContext()
start_time = time.time()
text_file = sc.textFile(input_file_name)
common_friends = get_common_friends(text_file)
common_friends.map(lambda x: ','.join(str(s) for s in x)).saveAsTextFile(output_file_name)
end_time = time.time()
comput_time = round(end_time - start_time, 3)
print("Nb of pairs generated: {}".format(common_friends.count()))
print("File saved at {}".format(output_file_name))
print("Computation for file: {}, took {} seconds.".format(file_name, comput_time))