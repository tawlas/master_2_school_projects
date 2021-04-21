#------------ Friends statistics
      
import sys
import os
from pyspark import SparkContext
import time

# Functions ---------------------------------------
def get_friends_stats(text_file):
    nb_friends_pairs = text_file.map(lambda x: x.split(",")).map(lambda x: len(x)-1)
    max_friends = nb_friends_pairs.reduce(lambda x,y: x if x > y else y)
    min_friends = nb_friends_pairs.reduce(lambda x,y: x if x < y else y)
    avg_friends = nb_friends_pairs.reduce(lambda x, y: x + y) / nb_friends_pairs.count()
    return ",".join([str(min_friends), str(max_friends), str(avg_friends)])

# Main code -------------------------------------------
input_path = "hdfs://sar01:9000/data/sn/"
output_path = "./"
file_name = os.path.basename(sys.argv[1])
input_file_name = input_path + file_name
output_file_name = output_path + "stats_" + os.path.splitext(file_name)[0]+".out"
# - create the Spark context and open the input file as a RDD (in the HDFS)
sc = SparkContext()
start_time = time.time()
text_file = sc.textFile(input_file_name)
stats_friends = get_friends_stats(text_file)
with open(output_file_name, 'w') as f:
      f.write(stats_friends)

end_time = time.time()
comput_time = round(end_time - start_time, 3)
print("File saved at {}".format(output_file_name))
print("Computation for file: {}, took {} seconds.".format(file_name, comput_time))