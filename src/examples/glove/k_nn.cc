/*
 * This program takes 2 files: one file contains list of seed users and other file contains list of universe of users
 * It outputs a file with expanded user list based on the given expansion factor(FACTOR variable as defined below).
 */

#include <falconn/lsh_nn_table.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>
#include <cmath>
#include <fstream>

#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <unordered_set>

#include <iostream>
using std::cerr;
using std::endl;
#include <fstream>
using std::ofstream;
#include <cstdlib>

using std::fmax;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;
using std::unordered_set;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::compute_number_of_hash_functions;
using falconn::SparseVector ;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::LSHNearestNeighborQuery;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::get_default_parameters;
typedef SparseVector<float,int> Point;
using falconn::core::CosineDistanceSparse;
CosineDistanceSparse<float> distance_function;
const string DATASET_FILE_NAME = "/mnt/lookalike/demograph_no_nulls.dat";
const string QUERY_FILE_NAME = "/home/vivek.vanga/FALCONN/src/examples/glove/dataset/query.dat";
const string EXPANSION_FILE_NAME = "/home/vivek.vanga/FALCONN/src/examples/glove/dataset/expanded1.txt";
const int SEED = 4057218;
const int NUM_HASH_TABLES = 8;
const int NUM_HASH_BITS = 14;
const int NUM_ROTATIONS = 2;
const int DATA_VECTOR_DIM = 4;
const int FEATURE_HASHING_DIMENSION=4;
const int FACTOR = 20;
const int NUM_OF_PROBES = 20;
/*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
bool read_point(FILE *file, Point *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  //std::vector<std::pair<float, int>> *  point_created = new std::vector<std::pair<float, int>>();
  int * int_buf = new int[d];
  if (fread(int_buf, sizeof(int), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  float * float_buf = new float[d];
  if (fread(float_buf, sizeof(float), d, file) != (size_t)d) {
      throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = std::make_pair(int_buf[i],float_buf[i]);
  }
  delete[] int_buf;
  delete[] float_buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, vector<Point> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  Point p;
  dataset->clear();
  while (read_point(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

/*
 * Normalizes the dataset.
 */
void normalize(vector<Point> *dataset) {
  for (size_t j=0; j < (*dataset).size();j++) {
    float denominator = 0.0;
    for(size_t i=0; i< (*dataset)[j].size();i++) {
      denominator += ((*dataset)[j][i].second) * ((*dataset)[j][i].second);
    }
    denominator = sqrt(denominator);
    if(denominator > 0) {
      for(size_t i=0; i< (*dataset)[j].size();i++) {
        (*dataset)[j][i].second /= denominator;
      }
    }
  }
}

/*
 * Computes the probability of success using a given number of probes.
 */
double evaluate_num_probes(LSHNearestNeighborTable<Point> *table,
                           const vector<Point> &queries,
                           const vector<int> &answers, int num_probes) {
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);
  int outer_counter = 0;
  int num_matches = 0;
  vector<int32_t> candidates;
  for (const auto &query : queries) {
    query_object->get_candidates_with_duplicates(query, &candidates);
    for (auto x : candidates) {
      if (x == answers[outer_counter]) {
        ++num_matches;
        break;
      }
    }
    ++outer_counter;
  }
  return (num_matches + 0.0) / (queries.size() + 0.0);
}

/*
 * Queries the data structure using a given number of probes.
 * It is much slower than 'evaluate_num_probes' and should be used to
 * measure the time.
 */
pair<double, QueryStatistics> evaluate_query_time(
    LSHNearestNeighborTable<Point> *table, const vector<Point> &queries,
    const vector<int> &answers, int num_probes) {
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);
  query_object->reset_query_statistics();
  int outer_counter = 0;
  int num_matches = 0;
  for (const auto &query : queries) {
    if (query_object->find_nearest_neighbor(query) == answers[outer_counter]) {
      ++num_matches;
    }
    ++outer_counter;
  }
  return make_pair((num_matches + 0.0) / (queries.size() + 0.0),
                   query_object->get_query_statistics());
}


int main() {
  try {
    vector<Point> dataset, queries;
    vector<int> answers;

    // read the dataset
    cout << "reading points" << endl;
    read_dataset(DATASET_FILE_NAME, &dataset);
    cout << dataset.size() << " dataset points read" << endl;

    read_dataset(QUERY_FILE_NAME, &queries);
    cout << queries.size() << "query points read" << endl;

    // normalize the data points
    cout << "normalizing points" << endl;
    normalize(&dataset);
    normalize(&queries);
    cout << "done" << endl;

    // find the center of mass
    Point * center = new Point();
    center->resize(DATA_VECTOR_DIM);
    for(int i=0;i<DATA_VECTOR_DIM;i++) {
      (*center)[i] = std::make_pair(i,0);
    }
    for (size_t i = 0; i < dataset.size(); ++i) {
      for(size_t j=0; j < dataset[i].size();j++) {
        int idx = dataset[i][j].first;
        float num = dataset[i][j].second;
        (*center)[idx].second = (*center)[idx].second + num;
      }
    }

    for (size_t i = 0; i < queries.size(); ++i) {
      for(size_t j=0; j < queries[i].size();j++) {
        int idx = queries[i][j].first;
        float num = queries[i][j].second;
        (*center)[idx].second = (*center)[idx].second + num;
      }
    }

    for(int i=0;i<DATA_VECTOR_DIM;i++) {
      (*center)[i].second /= (dataset.size() + queries.size());
    }

    // re-centering the data to make it more isotropic
    cout << "re-centering" << endl;
    for (size_t i = 0; i < dataset.size(); ++i) {
      for(size_t j=0; j < dataset[i].size();j++) {
        int idx = dataset[i][j].first;
          dataset[i][j].second -= (*center)[idx].second;
      }
    }

    for (size_t i = 0; i < queries.size(); ++i) {
      for(size_t j=0; j < queries[i].size();j++) {
        int idx = queries[i][j].first;
        queries[i][j].second -= (*center)[idx].second;
      }
    }

    // setting parameters and constructing the table
    //LSHConstructionParameters params;
   // params.dimension = dataset[0].size();
   // params.lsh_family = LSHFamily::CrossPolytope;
   // params.l = NUM_HASH_TABLES;
   // params.distance_function = DistanceFunction::EuclideanSquared;
   // compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
   // params.num_rotations = NUM_ROTATIONS;
    // we want to use all the available threads to set up
   // params.num_setup_threads = 0;
   // params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
   ///   For an easy way out, you could have used the following.

//    LSHConstructionParameters params
//      = get_default_parameters<Point>(dataset.size(),
//                                 972,
//                                 DistanceFunction::NegativeInnerProduct,
//                                 false);
//    cout << "building the index based on the cross-polytope LSH" << endl;
//    params.feature_hashing_dimension=256;

    LSHConstructionParameters params;
    params.dimension = DATA_VECTOR_DIM;
    params.lsh_family = LSHFamily::Hyperplane;
    params.l = NUM_HASH_TABLES;
    params.distance_function = DistanceFunction::EuclideanSquared;
    //params.distance_function = DistanceFunction::NegativeInnerProduct;
    compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
    cout << "num of hash functions";
    cout << params.k;
    params.num_rotations = NUM_ROTATIONS;
//    // we want to use all the available threads to set up
    params.num_setup_threads = 0;
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    params.feature_hashing_dimension = FEATURE_HASHING_DIMENSION;
//
//
    auto table = construct_table<Point>(dataset, params);
    cout << "table construction done" << endl;
    int num_probes = NUM_OF_PROBES;
    unordered_set<int> resultSet;
    int k = FACTOR;
    int idx = 0;
    for (const auto &query : queries) {
      cout << "new query: " << idx << " " << endl;
      idx+=1;
      int expansion = 0;
      int num_probes_for_this_query = num_probes;
      int k_for_query = k;
      do {
        expansion = 0;
        cout << "expansion reset. expansion: " << expansion << endl;
        std::vector<int> result;
        unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
                table->construct_query_object(num_probes_for_this_query);
        query_object->reset_query_statistics();
        query_object->find_k_nearest_neighbors(query,k_for_query,&result);
        cout << "result size: " << result.size() << endl;
        if(result.size() == 0) {
          cout << "doubling probes: " << num_probes_for_this_query << endl;
          num_probes_for_this_query = num_probes_for_this_query * 2;
          continue;
        }
        int overlap = 0;
        for(const auto res: result) {
          if(resultSet.find(res) == resultSet.end()  ) {
            expansion+=1;
          } else {
            overlap +=1;
          }
        }
        cout << "expansion: " << expansion << endl;
        cout << "overlap: " << overlap << endl;
        if(expansion >= FACTOR) {
          cout << "result set size: " << resultSet.size() << endl;
          int target = FACTOR;
          for(const auto res: result) {
            if(target == 0)
              break;
            resultSet.insert(res);
            target-=1;
          }
          cout << "result set size: " << resultSet.size() << endl;
        } else {
          num_probes_for_this_query = num_probes_for_this_query * 2;
          k_for_query = k_for_query * 2;
        }
      } while(expansion < FACTOR);
    }
    cout<< "query size: "<< queries.size();
    cout << "resultSet size: " << resultSet.size();
    std::vector<int> trimmedResultSet;
    for(const auto &res: resultSet) {
      trimmedResultSet.push_back(res);
      if(trimmedResultSet.size() == queries.size() * FACTOR)
        break;
    }
    cout << "trimmed ResultSet size: "<< trimmedResultSet.size() << endl;
    cout << "front" << trimmedResultSet.front() << endl;
    cout << "back" << trimmedResultSet.back() << endl;
    ofstream outdata;
    outdata.open(EXPANSION_FILE_NAME);
    for(const auto &res: trimmedResultSet) {
      outdata << res << endl;
    }
    outdata.close();
  } catch (runtime_error &e) {
    cerr << "Runtime error: " << e.what() << endl;
    return 1;
  } catch (exception &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "ERROR" << endl;
    return 1;
  }
  return 0;
}
